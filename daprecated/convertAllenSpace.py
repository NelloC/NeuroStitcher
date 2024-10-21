import re
import numpy

# unitCode: 
# - string that defines unit, with in-between brackets a multiplier in one or more dimensions.
#   The string can be m / mm / um. 
#   Examples: mm, mm(25), um(10,10,200)
# orientationCode:
# - set of three letters indicating the anatomical orientation of the positive x, y and z coordinate axes.
#   One axis must be assigned (R)ight or (L)eft,
#   another axis must be assigned (A)nterior or (P)osterior,
#   and the remaining axis must be assigned (S)uperior or (I)nferior.
#   Examples: PIR, LIP, RAS
# originCode:
# - string that indicates the landmark used as the origin.
#   For the Allen atlas volume, this can be 'center', 'corner', or 'ac (anterior commissure)'

def parseUnitCode(unitCode):
  m = re.match('(m|mm|um)(\(.*\))?$',unitCode)
  unit = m.group(1)
  group2 = m.group(2)
  multiplier = None
  if group2:
    multiplier = []
    m = re.match('\((.*)\)$',group2)
    group1 = m.group(1)
    groups = []
    if group1:
      groups = group1.split(',')
      for i,g in enumerate(groups):
        multiplier.append(float(g))
  return unit,multiplier

def parseOrientationCode(orientationCode):
  dims = None
  m = re.match('(R|L)(A|P)(S|I)$',orientationCode)
  if m:
    dims = [0,1,2]
  else:
    m = re.match('(R|L)(S|I)(A|P)$',orientationCode)
    if m:
      dims = [0,2,1]
    else:
      m = re.match('(A|P)(R|L)(S|I)$',orientationCode)
      if m:
        dims = [1,0,2]
      else:
        m = re.match('(A|P)(S|I)(R|L)$',orientationCode)
        if m:
          dims = [1,2,0]
        else:
          m = re.match('(S|I)(R|L)(A|P)$',orientationCode)
          if m:
            dims = [2,0,1]
          else:
            m = re.match('(S|I)(A|P)(R|L)$',orientationCode)
            if m:
              dims = [2,1,0]
  
  flip = None
  if m:
    flip = [0,0,0]
    targetOrientation = 'RAS'
    for i in range(0,3):
      flip[i] = 1 if m.group(1+i)==targetOrientation[dims[i]] else -1
  return dims,flip

def parseOriginCode(originCode,orientationCode):
  Annotation25_RAS_shape = numpy.array([456, 528, 320],float)
  targetOrigin_mm_RAS_center = Annotation25_RAS_shape/2*25e-3
  origin_mm_RAS = None
  m = re.match('center|corner|ac',originCode)
  if m:
    group = m.group(0)
    if group == 'center':
      origin_mm_RAS = targetOrigin_mm_RAS_center
    else:
      if group == 'ac':
        origin_voxel_RAS = numpy.array([228, 313, 113],float)
        origin_mm_RAS = origin_voxel_RAS*25e-3
      else: 
        if group == 'corner':
          dims,flip = parseOrientationCode(orientationCode)
          origin_mm_RAS = [0,0,0]
          for i in range(0,3):
            if flip[i]<0:
              origin_mm_RAS[dims[i]] = Annotation25_RAS_shape[dims[i]]*25e-3
  return origin_mm_RAS
      

def getAffine_unit(unit,multiplier):
  toMm = 1
  if unit == 'm': toMm = 1e3
  else:
    if unit == 'um': toMm = 1e-3
    
  if multiplier is None:
    multiplier = [1,1,1]
  else:
    numel = len(multiplier)
    if numel<3:
      lastValue = multiplier[numel-1]
      while numel<3:
        multiplier.append(lastValue)
        numel += 1
  return numpy.array([
    [toMm*multiplier[0], 0, 0, 0],
    [0, toMm*multiplier[1], 0, 0],    
    [0, 0, toMm*multiplier[2], 0],
    [0, 0, 0, 1]
  ],float)
  
def getAffine_orientation(dims,flip):
  A = numpy.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],    
    [0, 0, 0, 0],
    [0, 0, 0, 1]
  ],float)
  # A must be such that X_RAS = A*X
  A[dims[0],0] = flip[0]
  A[dims[1],1] = flip[1]
  A[dims[2],2] = flip[2]
  return A

def getAffine_origin(origin_mm_RAS):
  origin_mm_RAS_center = parseOriginCode('center','RAS')
  print('origin_mm_RAS_center',origin_mm_RAS_center)
  origin_shift = numpy.array(origin_mm_RAS,float)-numpy.array(origin_mm_RAS_center,float)
  print('origin_mm_RAS',origin_mm_RAS)
  return numpy.array([
    [1, 0, 0, origin_shift[0]],
    [0, 1, 0, origin_shift[1]],    
    [0, 0, 1, origin_shift[2]],
    [0, 0, 0, 1]
  ],float)
  
  
def toAllen_mm_RAS_center(unitCode,orientationCode,originCode):
  unit,multiplier = parseUnitCode(unitCode)
  A_unit = getAffine_unit(unit,multiplier)
  dims,flip = parseOrientationCode(orientationCode)
  A_reorient = getAffine_orientation(dims,flip)
  origin = parseOriginCode(originCode,orientationCode)
  A_origin = getAffine_origin(origin)
  return A_origin @ A_reorient @ A_unit
  
def test_toAllen_mm_RAS_center():
  unit,multiplier = parseUnitCode('m')
  print('unit, multiplier',unit,multiplier)
  assert(unit == 'm')
  assert(multiplier is None)
  unit,multiplier = parseUnitCode('mm(25)')
  print('unit, multiplier',unit,multiplier)
  assert(unit == 'mm')
  assert(len(multiplier) == 1 and multiplier[0] == 25)
  unit,multiplier = parseUnitCode('um(10,10,200)')
  print('unit, multiplier',unit,multiplier)
  assert(unit == 'um')
  assert(len(multiplier) == 3 and multiplier[0] == 10 and multiplier[1] == 10 and multiplier[2] == 200)
  A_unit = getAffine_unit(unit,multiplier)
  A_assert = numpy.array([
    [1e-3*10, 0, 0, 0],
    [0, 1e-3*10, 0, 0],    
    [0, 0, 1e-3*200, 0],
    [0, 0, 0, 1]
  ],float)
  assert(numpy.all(A_unit == A_assert))
  
  dims,flip = parseOrientationCode('PIR')
  print(dims,flip)
  assert(numpy.all(numpy.array(dims,int) == numpy.array([1,2,0],int)))
  A_reorient = getAffine_orientation(dims,flip)
  A_assert = numpy.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],    
    [0, -1, 0, 0],
    [0, 0, 0, 1]
  ],float)
  assert(numpy.all(A_reorient == A_assert))
  
  origin_mm_RAS = parseOriginCode('center','RAS')
  assert(numpy.all(numpy.array(origin_mm_RAS,float)==numpy.array([456*0.025/2, 528*0.025/2, 320*0.025/2],float)))
  A_translate = getAffine_origin(origin_mm_RAS)
  
  A = toAllen_mm_RAS_center('um','PIR','corner')
  
  A_allen2sba = numpy.array([
    [ 0.   ,  0.   ,  0.025, -5.7  ],
    [-0.025, -0.   , -0.   ,  5.35 ],
    [-0.   , -0.025, -0.   ,  5.15 ],
    [ 0.   ,  0.   ,  0.   ,  1.   ]
  ])
  print(A)
  
def convertAllenSpace(fromUnitOrientationOrigin=['um(25)','PIR','corner'],toUnitOrientationOrigin=['mm','RAS','ac']):
  toStandard = toAllen_mm_RAS_center(*fromUnitOrientationOrigin)
  toTarget = toAllen_mm_RAS_center(*toUnitOrientationOrigin)
  return numpy.linalg.inv(toTarget) @ toStandard
