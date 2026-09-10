### Ziffererkennung mit Neuronalen Netzen 
Dies ist ein Lehrnprjekt für mich selbst um die hinter Gründe neuronaler Netze besser zu verstehen. Als Wissen für den Ablauf der NN verwende ich die Artikel und Videos von Andrej Karpathy und die Harvard Introduction to computer science with python.
Die Backpropagation der NN ist selbst von mir implementiert. Es wurde lediglich mit numpy gearbeitet. Die Dateien visualisation.py und recognition.py sind nicht von mir selbst, sie dienen lediglich der Veranschaulichung der Funktion.
# Informationen zum NN 
Das NN soll handgeschriebene Ziffern erkennen können. Es wird mit dem Mnist Datenset trainiert. In dem Ornder variants kann man bereits trainierte Modelle finden.
Als Aktivierungs funktion wird Die ReLu Funktion verwendet. 
Kurz gefasst wird die Backpropagation errechnet, indem man den forwardpass mit den Ableitungen der Rechenoperatoren und der Aktivierungsfunktion zurückrechnet. Daher erhalt man das delta. Mit dem ehemaligen Weight - delta_weight * learning_rate, erhält man das neue, an den loss angepasste, Weight.
# Modelle feige1.0, feige1.1
Die beiden Modelle unterscheiden sich in Art wie sie funktoinieren kaum, das 1.0 besteht lediglich aus einem Hidden Layer und das 1.1 besteht aus zwei hidden layer. Backpropagation und forwardpass verlaufen nach dem genau gleichen System. Das Modell 1.1 ist aufgrund dem zweiten hidden Layer um 1 Prozent punkt besser in der Accuracy.
