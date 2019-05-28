from ..utils import protect

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QWidget, QLabel, QListWidget, QListWidgetItem,
                             QVBoxLayout, QFormLayout)

class QTensorflowInternals(QWidget):

    _graph = None
    
    _listTensors: QListWidget = None
    _labelNetworkType: QLabel = None

    def __init__(self, graph=None, **kwargs) -> None:
        """
        Arguments
        ---------
        parent: QWidget
        """
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()
        self.setGraph(graph)

    def setGraph(self, graph):
        self._graph = graph
        self._updateTensors()

    def _initUI(self) -> None:
        '''Initialize the user interface.'''
        self._labelNetworkType = QLabel()
        self._listTensors = QListWidget()
        self._listTensors.itemClicked.connect(self._onItemClicked)
        self._labelInputs = QLabel()
        self._labelOutputs = QLabel()
        self._labelOperation = QLabel()
        

    def _layoutUI(self) -> None:
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        layout.addWidget(self._listTensors)
        layout.addWidget(self._labelNetworkType)

        form = QFormLayout()
        form.addRow("Operation", self._labelOperation)
        form.addRow("Inputs", self._labelInputs)
        form.addRow("Outputs", self._labelOutputs)
        layout.addLayout(form)

    def _traverse(self, op):
        if op.name in self._nodes:
            return
        self._nodes[op.name] = (op.type == 'Placeholder')
        for i in op.inputs:
            iop = i.op
            self._traverse(iop)
            self._nodes[op.name] = self._nodes[op.name] or self._nodes[iop.name]
            
    def _updateTensors(self) -> None:
        self._listTensors.clear()

        if not self._graph:
            self._labelNetworkType.setText("")
            return  # nothing to do

        self._nodes = {}
        for op in self._graph.get_operations():
            self._traverse(op)


        for i, op in enumerate(self._graph.get_operations()):
            if not self._nodes[op.name]:
                continue

            inputs = ",".join([t.name for t in op.inputs])

            item = QListWidgetItem(f"({i}) {op.type}: '{op.name}'"
                                   f"({len(op.inputs)} inputs, "
                                   f"{len(op.outputs)} outputs)")
            item.setData(Qt.UserRole, op)
            if len(op.inputs) == 0 and len(op.outputs) == 0:
                item.setBackground(Qt.blue)
            if len(op.inputs) == 0:
                item.setBackground(Qt.green)
            elif len(op.outputs) == 0:
                item.setBackground(Qt.red)
            if op.type == 'Relu' or op.type == 'Softmax':
                item.setForeground(Qt.red)
            elif op.type == 'MaxPool' or op.type == 'Reshape':
                item.setForeground(QColor(255,165,0))  # orange
            elif op.type == 'LRN':  # local response normalization
                item.setForeground(Qt.blue)
            elif (op.type == 'Assign' or op.type == 'Identity' or
                  op.type == 'VariableV2'):
                # Variables
                item.setForeground(Qt.yellow)
                #item.setBackground(Qt.gray)
            self._listTensors.addItem(item)

        self._labelNetworkType.setText(str(self._listTensors.count()))

    @protect
    def _onItemClicked(self, item: QListWidgetItem) -> None:
        self._labelNetworkType.setText(item.text())
        op = item.data(Qt.UserRole)
        self._labelOperation.setText(f"{len(op.name)}")
        self._labelInputs.setText(", ".join([f"{t.name} [{t.shape}]" for t in op.inputs]))
        self._labelOutputs.setText(", ".join([f"{t.name} [{t.shape}]" for t in op.outputs]))
