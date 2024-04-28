# Data parameters for API

from pydantic import BaseModel

class Diamond(BaseModel):
    Shape :  str
    Weight :  float
    length :  float
    width :  float
    depth :  float
    Clarity :  str
    Colour :  str
    Cut :  str
    Polish :  str
    Symmetry :  str
    Fluorescence :  str