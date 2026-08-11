### Neural Network
Das ist ein Lehrn projekt um die systeme und mathematik dahinter besser zu versethen. Ich werde im folgenden Mathematische Konzepte für mich selbst und als kleine Mitschrift fals jemand anderes es sich ansehen möchte festhalten (:

# Daten einlesen und strukturieren 
im ersetn schritt werden die Daten der MNist Datenbank eingelesen und geteilt, einmal in X_train und Y_Train. Dabei ist y die Lösung, das heisst eine Zahl zwischen 0-9. Das X ist eine Liste aus den verschiedenen Pixeln, 728 Werte, jeder dieser werte geht von null bis 255.

# bias und weights startwerte festlegen
Als nächstes werden für jedes neuron ein startwert und bias festgelegt, in diesem fall einfach 0. Ich arbeite hier mit einem hidden layer mit jeweils 128 parametern.

# ReLu Aktivierungsfunktion
durch die ReLu Aktivierungsfunktion wir das linearitäts Problem gelöst. Jedes Ergebniss das negativ ist wird automatisch zu null. Es wird nach der aufsummierung der weights und werte angewendet.

# Softmax Algorythmus
In Softmax wird zuerst dafür gesorgt das die float werte nicht zu inf werden, dafür wird der höchste wert aller klassen von allen anderen abgezogen. Dabei entstehen auch negative Werte, diese negativen Werte werden durch das Eulersche Verfahren wieder in Relateion zueinander gesetzt. Am schluss werden die werte in eine Wahrscheinlichkeit zwischen 0 - 1 durch sich selbst als ganzes geteilt.

# Forward Funktion
Die Forward funktion ist zuerst für die aufsummierung der Produkte aus werte und bias zuständig, und bias addieren, danach wird im 128 parameter hidden layer die ReLu Funkion angewendet um die linearität loszuwerden. Danach wird im output layer wieder aufsummiert und am ende softmax angewandt um eine Wahrscheinlichkeit für ein Ergebniss zu erhalten.

# test