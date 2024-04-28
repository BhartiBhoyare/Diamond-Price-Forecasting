import pickle 
import pandas as pd

#test DataFrame
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
Shape = "CUSHION"
Weight = "0.5"
length = "4.77"
width = "4.38"
depth = "2.89"
Clarity = "VS2"
Colour = "F"
Cut = "VG"
Polish = "EX"
Symmetry = "VG"
Fluorescence = "N"

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
test_data[ClarityC] = True
test_data[ColourC] = True
test_data[CutC] = True
test_data[PolishC] = True
test_data[SymmetryC] = True
test_data[FluorescenceC] = True


#our diamond pridiction model
pickle_model = pd.read_pickle(r'./model.pkl')


predicted = pickle_model.predict(test_data)

print(predicted)