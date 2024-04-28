import uvicorn 
from data_formate import Diamond
from fastapi import FastAPI
import pandas as pd
from starlette.middleware.cors import CORSMiddleware 

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pickle_model = pd.read_pickle(r'./model.pkl')

@app.get('/')   #routes, it give data to website
def index():
    return {"Hello Diamonds"}

@app.post('/predict')    #routes, it get data from website
def predict(data:Diamond):

    data = data.dict()

    test_data = pd.DataFrame({
  "Weight": [0],
  "length": [0],
  "width": [0],
  "depth": [0],
  "Shape_CUSHION": [False],
  "Shape_EMERALD": [False],
  "Shape_HEART": [False],
  "Shape_MARQUISE": [False],
  "Shape_OVAL": [False],
  "Shape_PEAR": [False],
  "Shape_PRINCESS": [False],
  "Shape_RADIANT": [False],
  "Shape_ROUND": [False],
  "Clarity_FL": [False],
  "Clarity_I1": [False],
  "Clarity_I2": [False],
  "Clarity_I3": [False],
  "Clarity_IF": [False],
  "Clarity_SI1": [False],
  "Clarity_SI2": [False],
  "Clarity_VS1": [False],
  "Clarity_VS2": [False],
  "Clarity_VVS1": [False],
  "Clarity_VVS2": [False],
  "Colour_D": [False],
  "Colour_E": [False],
  "Colour_F": [False],
  "Colour_FANCY": [False],
  "Colour_FBG": [False],
  "Colour_G": [False],
  "Colour_H": [False],
  "Colour_I": [False],
  "Colour_J": [False],
  "Colour_K": [False],
  "Colour_L": [False],
  "Colour_M": [False],
  "Colour_N": [False],
  "Colour_O": [False],
  "Colour_O-P": [False],
  "Colour_Q": [False],
  "Colour_Q-R": [False],
  "Colour_S-T": [False],
  "Colour_U-V": [False],
  "Colour_W": [False],
  "Colour_W-X": [False],
  "Colour_Y-Z": [False],
  "Cut_EX" : [False] ,
  "Cut_F" : [False] ,
  "Cut_GD" : [False] ,
  "Cut_VG" : [False] ,
  "Polish_EX" : [False] ,
  "Polish_F" : [False] ,
  "Polish_GD" : [False] ,
  "Polish_VG" : [False] ,
  "Symmetry_EX" : [False],
  "Symmetry_FR" : [False],
  "Symmetry_GD" : [False],
  "Symmetry_VG" : [False],
  "Fluorescence_F" : [False],
  "Fluorescence_M" : [False],
  "Fluorescence_N" : [False],
  "Fluorescence_SL" : [False],
  "Fluorescence_ST" : [False],
  "Fluorescence_VS" : [False],
  "Fluorescence_VSL" : [False]
        })


    # input data
    Shape = data['Shape']
    Weight = data['Weight']
    length = data["length"]
    width = data["width"]
    depth = data["depth"]
    Clarity = data["Clarity"]
    Colour = data["Colour"]
    Cut = data["Cut"]
    Polish = data["Polish"]
    Symmetry = data["Symmetry"]
    Fluorescence = data["Fluorescence"]


    ClarityC = "Clarity_" + Clarity
    ColourC = "Colour_" + Colour
    CutC = "Cut_" + Cut
    PolishC = "Polish_" + Polish
    SymmetryC = "Symmetry_" + Symmetry
    FluorescenceC = "Fluorescence_" + Fluorescence
    ShapeC = "Shape_" + Shape

    test_data[ShapeC] = True
    test_data['Weight'] = Weight
    test_data['length'] = length
    test_data['width'] = width
    test_data['depth'] = depth
    test_data[ClarityC ] = True
    test_data[ColourC] = True
    test_data[CutC] = True
    test_data[PolishC] = True
    test_data[SymmetryC] = True
    test_data[FluorescenceC] = True

    return pickle_model.predict(test_data)[0]


if __name__ == '__main__':
    uvicorn.run(app,port=8000)

# uvicorn main:app --reload --port 8000 (to run this api)