"""
This module provides functions for processing modifications and
extracting modification related information.

"""
import dataclasses

from typing import Optional

from pepfrag import ModSite


@dataclasses.dataclass(frozen=True)
class Mod(ModSite):

    @property
    def int_site(self) -> int:
        """ Converts site to integer so that the residue can be extracted
        using seq[site-1]

        returns:
            Integer site.
        """
        if isinstance(self.site, int):
            return self.site
        if self.site == "nterm":
            return 1
        # Extract C-terminus residue when using `site - 1`
        return 0

    @property
    def name(self) -> Optional[str]:
        """ Valid modification name.

        Returns:
            Modification name if it's not `unknown`, otherwise None
        """
        return None if self.mod == "unknown" else self.mod
