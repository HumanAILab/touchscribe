def get_direction_by_degree(d):
    output=[]
    if d >= 345 and d <= 360 or d >=0 and d < 15: output.append(12)
    elif d >=15 and d < 45: output.append(1)
    elif d >=45 and d < 75: output.append(2)
    elif d >=75 and d < 105:output.append(3)
    elif d >=105 and d < 135: output.append(4)
    elif d >=135 and d < 165: output.append(5)
    elif d >=165 and d < 195: output.append(6)
    elif d >=195 and d < 225: output.append(7)
    elif d >=225 and d < 255: output.append(8)
    elif d >=255 and d < 285: output.append(9)
    elif d >=285 and d < 315: output.append(10)
    elif d >=315 and d < 345: output.append(11)
    
    return output[0]