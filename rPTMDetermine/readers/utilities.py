import os
from typing import Dict, Generator


def reverse_readline(
        filename: str,
        buf_size: int = 8192
) -> Generator[str, None, None]:
    """
    A generator that returns the lines of a file in reverse order.

    Based on https://stackoverflow.com/questions/2301789/.

    """
    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # The first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # If the previous chunk starts right from the beginning of line
                # do not concat the segment to the last line of new chunk.
                # Instead, yield the segment first
                if buffer[-1] != '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if lines[index]:
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment


def xml_line_to_dict(line: str) -> Dict[str, str]:
    """
    Parses a line of XML to a dictionary of attribute key/value pairs.

    """
    line = line.strip()
    attributes = {}
    for element in line.split(' '):
        if '=' in element:
            key, value = element.split('=')
            attributes[key] = value.strip('"')
    return attributes
