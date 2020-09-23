#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demo program to test widget of the Qt GUI.



"""

# standard imports
import sys
import importlib
from argparse import ArgumentParser

# third party imports
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout

# toolbox imports
from dltb.util.image import imread

widget_names = {
    'QImageView': 'qtgui.widgets.image.QImageView',
    'QNetworkComboBox': 'qtgui.widgets.network.QNetworkComboBox',
    'QDatasourceComboBox': 'qtgui.widgets.datasource.QDatasourceComboBox'
}


def main() -> None:

    parser = ArgumentParser(description='QtGUI demo program.')
    parser.add_argument('-i', '--image', action='append', nargs='+',
                        help='an image to be used')
    parser.add_argument('-n', '--network',
                        help='network to be used')
    parser.add_argument('-t', '--toolbox', action='store_true',
                        help='use a Toolbox object')
    parser.add_argument('-w', '--window', action='store_true',
                        help='create a QMainWindow')
    parser.add_argument('widget', nargs='+',
                        help='widgets to display')

    print(sys.argv)
    args = parser.parse_args()
    print(args)
    # args, unknown_args = parser.parse_known_args(sys.argv)
    # print(args, unknown_args)

    images = ([image for sublist in args.image for image in sublist]
              if args.image else
              ['/space/data/ibug300/300W/01_Indoor/indoor_034.png'])  # FIXME[hack]

    widgets = [widget_names.get(name, name).rsplit('.', 1)
               for name in args.widget]
    print(widgets)

    app = QApplication(sys.argv[:1])
    if args.window:
        window = QMainWindow()
        centralWidget = QWidget()
        layout = QVBoxLayout()
        centralWidget.setLayout(layout)
        window.setCentralWidget(centralWidget)
        window.show()

    for module_name, cls_name in widgets:
        module = importlib.import_module(module_name)
        Widget = getattr(module, cls_name)
        qwidget = Widget()
        if cls_name == 'QImageView':
            qwidget.setImage(imread(images[0]))
        elif cls_name == 'QNetworkComboBox':
            qwidget.setOnlyInitialized(False)
        elif cls_name == 'QDatasourceComboBox':
            qwidget.setOnlyInitialized(False)

        if args.window:
            layout.addWidget(qwidget)
        else:
            qwidget.show()

    app.exec_()


if __name__ == '__main__':
    main()
