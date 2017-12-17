Ideen
=====

* Doku + Aufr&uml&auml;umen
* Neues Networkinterface in GUI verwenden
* Bugs fixen, Unittests
* Daten laden, Wie Inputs falscher Gr&ouml;&szlig;e handhaben?
* Netze laden, verschiedene Formate
* Code von Ulf aufräumen, Modularisierung, Refactoring
* Models, Algorithmen, Interfaces (GUI, CLI)
* Wie genau Aktivierung zur&uml;ck propagieren (deepvis-toolbox)
* Heatmapping
* Verschiedene Nutzerschnittstellen
* Filterkernel f&uml;r jede Convloutional layer
* Bei h&ouml;hren Layers die einzelnen Kernel f&uml;r jeden Inputchannel anzeigen
* Klassifikation (richtig/falsch f&uml;r aktuellen Input anzeigen)
* Rezeptive felder einzelner Conv-Neurouns anzeigen
* Article about feature visualization: https://news.ycombinator.com/item?id=15646037
* Generative Models visualisieren durch Wahl von Inputfeatures

x. Maybe ONNX support?



Fragen
======

1. Wie kann ich mit qstat Informationen über spezifische queues erhalten?
    - `qstat -q cv.q` oder ähnliches gibt nichts
2. Wie beantrage ich Cuda?
3. Wo finde ich Dokumentation?


Wtf?
====

`panels/occlusion.py` and others are unused?

Refactoring
===========

Main parts
----------
* Networks
* Model
    -   Initialised with network
    -   Contains current state of which layer/unit selected
    -   Can be queried for layer/unit
    -   can be given new data set
* Algorithms
    -   used by model to generate overlays for units or other elements
    -   Is given a network + data by the model
* View
    -   can be a gui or command line interface
    -   pulls new data from the model when prompted
    -   delegates incoming events to the controller
* Controller
    -   Comes in GUI or CMDLine flavor
    -   Receives user event (clicks, keypresses, commands)
    -   Informs GUI that it should pull new data from the model
    -

!! Do all expensive computations in model asynchronously
