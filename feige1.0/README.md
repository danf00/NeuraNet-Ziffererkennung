### Ziffererkennung mit Neuronalen Netzen 
Dies ist ein Lehrnprjekt für mich selbst um die hinter Gründe neuronaler Netze besser zu verstehen. 
Die Backpropagation der NN ist selbst von mir implementiert. Es wurde lediglich mit numpy gearbeitet. Die Dateien visualisation.py und recognition.py sind nicht von mir selbst, sie dienen lediglich der Veranschaulichung der Funktion.
# Inforamtionen zum NN 
Das NN soll handgeschriebene Ziffern erkennen können. Es wird mit dem Mnist Datenset trainiert. In dem Ornder variants kann man bereits trainierte Modelle finden.
Das NN besteht aus Input-Layer, 1 hidden Layer, und einem output Layer. 
Es hat 784 Inputs, 128 neuronen im Hidden Layer und 10 output neuronen. 
Als Aktivierungs funktion wird Die ReLu Funktion verwendet. 
Kurz gefasst wird die Backpropagation errechnet, indem man den forward pass mit den Ableitungen der Rechenoperatoren und der Aktivierungsfunktion zurückrechnet. Daher erhalt man das delta. Mit dem ehemaligen Weight - delta_weight * learning_rate, erhält man das neue, an den loss angepasste, Weight. 