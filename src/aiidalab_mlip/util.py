"""Contains utility functions used throughout the python package."""

from collections.abc import Callable, Collection, Iterable
from itertools import count, islice
from typing import Protocol, TypeVar, overload

import ipywidgets as ipw
from IPython.display import Javascript, display

AT = TypeVar("AT", ipw.Accordion, ipw.Tab)


class SupportsStr(Protocol):
    """Any class which supports __str__ method."""

    def __str__(self) -> str: ...


class SupportsRepr(Protocol):
    """Any class which supports __repr__ method."""

    def __repr__(self) -> str: ...


SupportsFormat = SupportsStr | SupportsRepr


def open_link_in_new_tab(path: str, _=None) -> None:
    """
    Open a given link in a new browser tab.

    Parameters
    ----------
    path :  str
        The link to be opened.
    """
    js_code = f"window.open('{path}', '_blank');"
    display(Javascript(js_code))


def tab_from_dict(typ: type[AT], tabs: dict[str, ipw.Widget]) -> AT:
    """Create a tab or accordion from a dict.

    Dict maps titles to widgets.

    Parameters
    ----------
    typ : type[ipw.Accordion | ipw.Tab]
        Accordion or Tab to create.
    tabs : dict[str, ipw.Widget]
        Dict of tabs to widgets.

    Returns
    -------
    Accordion or Tab
        Constructed Accordion or Tab.
    """
    blk = typ(list(tabs.values()))
    for ind, name in enumerate(tabs):
        blk.set_title(ind, name)

    return blk


@overload
def get_next_name(
    template: str,
    *,
    exists: Collection[str] | Callable[[str], bool],
    trial: Iterable[SupportsFormat] | None = ...,
    max_tries: int | None = ...,
    default: None = ...,
    **kwargs: SupportsFormat,
) -> str | None: ...
@overload
def get_next_name(
    template: str,
    *,
    exists: Collection[str] | Callable[[str], bool],
    trial: Iterable[SupportsFormat] | None = ...,
    max_tries: int | None = ...,
    default: str = ...,
    **kwargs: SupportsFormat,
) -> str: ...
def get_next_name(
    template: str,
    *,
    exists: Collection[str] | Callable[[str], bool],
    trial: Iterable[SupportsFormat] | None = None,
    max_tries: int | None = None,
    default: str | None = None,
    **kwargs: SupportsFormat,
) -> str | None:
    """Return the first unused name given rules for next in sequence and invalid values.

    Parameters
    ----------
    template : str
        Base format string to modify, must contain ``trial`` field.
    exists : Collection[str] | Callable[[str], bool]
        Set of existing values to skip or :ref:`Callable` determining existance.
    trial : Iterable[SupportsFormat], optional
        Set/Generator of trial values to use. If ``None`` defaults to :ref:`itertools.count`.
    max_tries : int, optional
        Number of attempts to generate, unlimited if ``None``.
    default : str, optional
        Default to return if ``max_tries`` reached or ``trial`` exhausted.
    **kwargs : SupportsFormat
        Extra substitutions to pass into template.

    Returns
    -------
    str | None
        Next unused name.

    Notes
    -----
    The special value which is substituted is named "trial".

    Examples
    --------
    >>> tpl = "hello_{trial}"
    >>> get_next_name(tpl, exists=())
    'hello_1'
    >>> get_next_name(tpl, exists=lambda x: x[-1] != "6")
    'hello_6'
    >>> get_next_name(tpl, exists={"hello_1", "hello_2"})
    'hello_3'
    >>> get_next_name(tpl, exists={"hello_1", "hello_2"}, max_tries=1, default="Argh!")
    'Argh!'
    >>> get_next_name("{a}_hello_{trial}", exists=(), a="big")
    'big_hello_1'
    """
    if trial is None:
        trial = count(1)

    if isinstance(exists, Collection):
        exists = exists.__contains__

    gen = (template.format(trial=elem, **kwargs) for elem in trial)
    return next((x for x in islice(gen, max_tries) if not exists(x)), default)
