# Coding Standards for the Deep Learning ToolBox


## Attribute or property?


## Property or methods?

Python allows to define dynamic properties using the `@property`
decorator.  This allows to replace classical getter and setters by a
more natural looking alternative.  Properties can also be read-only
(only a getter; and in theory alse write only). 

Formally, a read-only property seems to be equivalent to a method with
no arguments, that returns some value. So the question arises, when to
use what?

```python
# classical approach using methods
player.start()
player.is_running()  # True
player.stop()
player.is_running()  # False

# same idea, using properties
player.running = True       # setting property to True starts the player
player.running = False      # setting property to False stops the player
```

So criteria for helping in deciding between property or method could
be the following:

* Is the value mainly intended as a storage location?
  If yes, a property seems the more natural choice.  It may
  even be realized by an attribute and replaced by a property
  later, if more control is desired.

* Can the value be influenced by processes outside the class,
  that is without calling methods of that class?
  E.g. does the value depend on the availability of some resource?
  If yes, a method seems to be the better choice, as it makes
  clear that the value is recomputed and hence can change on
  every call.

* Does setting the value start or stop some process?
  If yes, a method seem to be the better choice.
  A method also allows to pass further parameters to
  configure the process.


## Asynchrony


# Pylinting the Deep Learning ToolBox

```
pylint
```

## Pylinting the QtGUI


## Known problems


# Typechecking the Deep Learning ToolBox

```
mypy
```

## Known problems



# Testing the Deep Learning ToolBox

```
pytest
```
