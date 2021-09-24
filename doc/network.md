



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



# Pretrained networks

# Third-party models

Most deep learning libraries provide predefined models as well
pretrained weights for this models.


## Torch: Torch.hub

* Torch: [torch.hub](https://pytorch.org/docs/stable/hub.html)


