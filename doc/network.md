



# Should `Network`s be `Tool`s

Pro:
* some `Network`s realize `Tool`, there are for example
  `Network`s that can be used as `SoftClassifier`.
  

Contra:
* there is some conflicts with meta classes: both `Network`
  and `Tool` are `RegisterClass`es.
* both may want to implement a functional API


(Preliminary) conclusion:
* Create a wraper `Tool` that can apply a network to do some task.

