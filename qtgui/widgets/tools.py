"""
.. moduleauthor:: Ulf Krumnack

.. module:: qtgui.widgets.tools

This module contains widgets for viewing and controlling
:py:class:`Tools`s. It aims at providing support for all
abstract interfaces defined in `tools`.
"""

# standard imports
import logging

# Qt imports
from PyQt5.QtCore import pyqtSignal

# toolbox imports
from toolbox import Toolbox
from dltb.tool import Tool

# GUI imports
from ..utils import protect
from .register import QRegisterListWidget, QRegisterComboBox
from .register import ToolboxAdapter

# logging
LOG = logging.getLogger(__name__)


class ToolAdapter(ToolboxAdapter, qobservables={
        Toolbox: {'tool_changed', 'tools_changed'}}):
    # pylint: disable=abstract-method
    """A :py:class:`ToolboxAdapter` that is especially interested in
    :py:class:`Tool`.

    There are different ways to use a :py:class:`ToolItemList`:
    * Standalone: selectable :py:class:`Tool`\\ s have to be
      added and removed explicitly by calling
      :py:meth:`addTool` and :py:class:`removeTool`.
    * Directly observing the py:class:`Tool` register. The
      :py:class:`ToolItemList` will be updated when new tools
      are initialized.
    * Based on a :py:class:`Toolbox`: tools registered with the
      :py:class:`Toolbox` can be selected. The :py:class:`ToolItemList`
      will be updated when tools are added or removed from the
      :py:class:`Toolbox`.

    A :py:class:`ToolItemList` can be configured to only show a
    sublist of the actual tool list. This is useful when run
    in Register or Toolbox mode, but only a subset of the available
    tools shall be presented:
    * preparation: just show tools that are prepared
    * superclass: just show tools that are subclasses of a given
      superclass, e.g., only face tools.

    Attributes
    ----------
    _toolbox: Toolbox
        A :py:class:`Toolbox` providing a list of
        :py:class:`Tool`\\ s. Can be set with :py:meth:`setToolbox`,
        making the :py:class:`ToolItemList` an observer of that
        :py:class:`Toolbox`, reacting to 'tools_changed' signals.

    """

    toolSelected = pyqtSignal(object)

    def updateFromToolbox(self) -> None:
        """Update the list from the :py:class:`Toolbox`.
        """
        self.updateFromIterable(self._toolbox.tools)

    def toolbox_changed(self, _toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        # pylint: disable=invalid-name
        """React to a change in the :py:class:`Toolbox`. The only change
        of interest is a change of the current tool. This
        will be reflected in the list.
        """
        if change.tool_changed:  # the current tool has changed
            self._formatAllItems()
        elif change.tools_changed:  # the list of tools has changed
            self.updateFromToolbox()

    def currentTool(self) -> Tool:
        """The currently selected Tool in this
        :py:class:`ToolAdapter`.
        """
        item = self._currentItem()
        if self._toolbox is not None:
            # items are of type Tool
            return item

        # items are of type InstanceRegisterEntry
        return None if item is None else item.obj


class QToolListWidget(ToolAdapter, QRegisterListWidget):
    """A list displaying the :py:class:`Tool`s of a
    :py:class:`Toolbox`.

    By providing a :py:class:`Toolbox`, the list becomes clickable,
    and selecting a Tool from the list will set current tool of
    the :py:class:`Toolbox`, and vice versa, i.e. changing the current
    tool in the :py:class:`Toolbox` will change the current item in
    the list.

    """

    def __init__(self, **kwargs) -> None:
        """
        """
        super().__init__(register=Tool.instance_register, **kwargs)

    @protect
    def _oncurrentIndexChanged(self, index: int) -> None:
        """A forward to map item selection to Tool selection.
        """
        self.toolSelected.emit(self.itemData(index))


class QToolComboBox(ToolAdapter, QRegisterComboBox):
    """A widget to select a :py:class:`tool.Tool` from a list of
    :py:class:`tool.Tool`s.


    The class provides the common :py:class:`QComboBox` signals, including

    activated[str/int]:
        An item was selected (no matter if the current item changed)

    currentIndexChanged[str/int]:
        An new item was selected.

    """

    def __init__(self, **kwargs) -> None:
        """
        """
        super().__init__(register=Tool.instance_register, **kwargs)
        self.currentIndexChanged.connect(self._oncurrentIndexChanged)

    @protect
    def _oncurrentIndexChanged(self, _index: int) -> None:
        """A forward to map item selection to Tool selection.
        """
        self.toolSelected.emit(self.currentTool())
