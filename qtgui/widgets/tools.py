"""
.. moduleauthor:: Ulf Krumnack

.. module:: qtgui.widgets.tools

This module contains widgets for viewing and controlling
:py:class:`Tools`s. It aims at providing support for all
abstract interfaces defined in `tools`.
"""

# standard imports
from typing import Iterator
import logging

# Qt imports
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QHideEvent, QKeyEvent
from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel

# toolbox imports
from toolbox import Toolbox
from tools import Tool

# GUI imports
from ..utils import QObserver, protect
from .register import QRegisterList, QRegisterController, QPrepareButton
from .register import RegisterItemList, QRegisterItemComboBox
from .navigation import QIndexControls as QBaseIndexControls

# logging
LOG = logging.getLogger(__name__)


class ToolItemList(RegisterItemList, qobservables={
        Toolbox: {'tools_changed', 'tool_changed'}}):
    """There are different ways to use a :py:class:`ToolItemList`:
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
        :py:class:`Tool`\ s. Can be set with :py:meth:`setToolbox`,
        making the :py:class:`ToolItemList` an observer of that
        :py:class:`Toolbox`, reacting to 'tools_changed' signals.

    """

    def __init__(self, toolbox: Toolbox = None,
                 tool: Tool = None, **kwargs) -> None:
        """Initialization of the :py:class:`ToolItemList`.

        Parameters
        ----------
        """
        super().__init__(register=Tool, **kwargs)
        self.setToolbox(toolbox)
        self.setTool(tool)

    def setToolbox(self, toolbox: Toolbox) -> None:
        """Set a :py:class:`Toolbox` from which a list of tools can
        be obtained.
        """
        # palette = QPalette();
        # palette.setColor(QPalette.Background, Qt.darkYellow)
        # self.setAutoFillBackground(True)
        # self.setPalette(palette)
        return
        if toolbox is not None:
            self._updateFromIterator(toolbox.tools)
        else:
            self._updateFromIterator(iter(()))

    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        """React to changes in the Toolbox. We are only interested
        when the list of tools has changed, in which case we will
        update the content of the QComboBox to reflect the available
        Tools.
        """
        if change.tools_changed:
            self._updateFromIterator(toolbox.tools)

    def currentTool(self) -> Tool:
        """Get the currently selected :py:class:`Tool`.
        This may be `None` if no tool is selected.
        """
        return self.currentItem()

    def setCurrentTool(self, tool: Tool) -> None:
        """Select the given :py:class:`Tool`.

        Arguments
        ---------
        tool: Tool
            The tool to become the currently selected element
            in this list. `None` will deselect the current element.

        Raises
        ------
        ValueError:
            The given :py:class:`Tool` is not an element of this
            :py:class:`ToolItemList`.
        """
        self.setCurrentItem(tool)

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events. The :py:class:`QImageView` supports
        the following keys:

        r: toggle the keepAspectRatio flag
        """
        key = event.key()
        LOG.debug("ToolItemList.keyPressEvent: key=%d", key)

        if key == Qt.Key_T:  # t = debug toolbox:
            if self._toolbox is None:
                print("No Toolbox")
            else:
                print("Toolbox tools:")
                for tool in self._toolbox.tools:
                    print("tool:", tool)
        else:
            super().keyPressEvent(event)


class QToolSelector(QRegisterItemComboBox, ToolItemList,
                          qobservables={Tool: {'state_changed'}}):
    """A widget to select a :py:class:`Tool` from a list of
    :py:class:`Tool`\ s.

    """
    toolSelected = pyqtSignal(object)

    def __init__(self, **kwargs) -> None:
        """Initialization of the :py:class:`QToolSelector`.

        Parameters
        ----------
        """
        super().__init__(**kwargs)
        self.currentIndexChanged.connect(self._oncurrentIndexChanged)

    @protect
    def _oncurrentIndexChanged(self, index: int) -> None:
        """A forward to map item selection to Tool selection.
        """
        self.toolSelected.emit(self.itemData(index))

    def tool_changed(self, tool: Tool, change: Tool.Change) -> None:
        """React to a change of an observed tool.
        """
        LOG.debug("QToolSelector: tool %s changed %s.",
                  tool, change)
        self._updateItems()

    def _addItem(self, item: Tool) -> None:
        """Add an item to this :py:class:`QRegisterItemComboBox`.
        It is assumed that the item is not yet contained in this
        :py:class:`QRegisterItemComboBox`.
        """
        super()._addItem(item)
        self.observe(item, interests=Tool.Change('state_changed'))

    def _removeItem(self, item: Tool) -> None:
        """Remove an item from this :py:class:`QRegisterItemComboBox`.  It is
        assumed that the item is contained in this
        :py:class:`QRegisterItemComboBox`, otherwise a
        :py:class:`ValueError` is raised.

        """
        self.unobserve(item)
        super()._removeItem(item)
