

# Data

Data is a subclass of Observable (Changeable)
* attribute_added
* attribute_changed

Data - Datalike
Image - Imagelike
Sound - Soundlike




# Datasource

* Datasource is `Observable`, Stateful (prepared/unprepared, busy, failed)

## The get API

## The context manager API



## Availabe Datasource

* Iterable: __iter__
* Indexed: __len__, __getitem__
* Life: awake, kill

* Imagesource
* Soundsource

* Directory: Indexed
* Array: Indexed
* Webcam: Iterable, Life
* Video: Indexed, Life
* Random: Life



# Datafetcher

* A `Datafetcher` can fetch from a `Datasource`
* A `Datafetcher` is `Observable`, it can notify observers once
  data is fetched.


### The API

* `fetch(...)`
* `loop(...)`


