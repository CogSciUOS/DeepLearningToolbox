from PyQt5.QtWidgets import QWidget, QLabel, QGridLayout
from observer import Observer
from model import Model


class QClassesView(QWidget, Observer):
    '''QClassesView is a QWidget to display classification results.
    It is intended to be used with networks that act as classifier.
    '''

    _top_n = 5
    _classes = []
    _scores = []
    
    def __init__(self, parent: QWidget=None, top_n=5):
        '''Initialization of the QActivationView.

        Parameters
        -----------
        parent  :   QtWidget
                    Parent widget (passed to super)
        '''
        super(QClassesView, self).__init__(parent)
        self._top_n = top_n
        self._initUI()

    def _initUI(self):
        layout = QGridLayout()
        for i in range(self._top_n):
            self._classes.append(QLabel("X", self))
            layout.addWidget(self._classes[i], i, 0)
            self._scores.append(QLabel("Y", self))
            layout.addWidget(self._scores[i], i, 1)
            print("adding labels")
        self.setLayout(layout)

    def modelChanged(self, model, info):
        '''The QClassesView is only interested if the classification result
        changes.
        '''
        if info.activation_changed:
            top_n_scores = model.top_n_classifications(self._top_n)
            print("QClassesView: model Changed:", top_n_scores)

            if top_n_scores is not None:
                for i, (c, s) in enumerate(top_n_scores.items()):
                    self._classes[i].setText(str(c))
                    self._scores[i].setText(str(s))
            else:
                for i in range(self._top_n):
                    self._classes[i].setText("None")
                    self._scores[i].setText("None")
            self.update()
