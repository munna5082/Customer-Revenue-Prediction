from django.shortcuts import render
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import model_from_json

with open("crpmodel.json", "r")as file:
    raw_model = file.read()

model = model_from_json(raw_model)
model.load_weights("model.h5")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# Create your views here.
def base(request):
    res = None
    result = ""
    if request.method == 'POST':
        if request.POST.get('pred_button'):
            name = request.POST['Product Name']
            Administrative = request.POST['Administrative']
            Informational = request.POST['Informational']
            ProductRelated = request.POST['ProductRelated']
            BounceRates = request.POST['BounceRates']
            ExitRates = request.POST['ExitRates']
            PageValues = request.POST['PageValues']
            SpecialDay = request.POST['SpecialDay']
            OperatingSystems = request.POST['OperatingSystems']
            Browser = request.POST['Browser']
            Region = request.POST['Region']
            TrafficType = request.POST['TrafficType']
            VisitorType = request.POST['VisitorType']
            weekend = request.POST['weekend']

            # print(name, type(Administrative))
            if name != "":
                try:
                    df = pd.DataFrame(columns=['Administrative_Duration', 'Informational_Duration', 'ProductRelated',
                                                'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
                                                'OperatingSystems', 'Browser', 'Region', 'TrafficType',
                                                'Weekend', 'New_Visitor', 'Other', 'Returning_Visitor'])

                    type_visitor = finding_type(VisitorType)
                    df2 = {'Administrative_Duration': float(Administrative), 'Informational_Duration': float(Informational),
                           'ProductRelated': float(ProductRelated), 'BounceRates': float(BounceRates), 'ExitRates':
                            float(ExitRates), 'PageValues': float(PageValues), 'SpecialDay': float(SpecialDay),
                            'OperatingSystems': int(OperatingSystems), 'Browser': int(Browser), 'Region': int(Region),
                            'TrafficType': int(TrafficType), 'Weekend': int(weekend), 'New_Visitor':
                            type_visitor[0], 'Other': type_visitor[1], 'Returning_Visitor': type_visitor[2]}

                    df = df.append(df2, ignore_index=True)

                    # predict the output
                    pred = model.predict(df)
                    res = (pred>0.5).astype(int)
                    print(res)
                    if res[0] == 1:
                        result = "True"
                    else:
                        result = "False"
                except:
                    pass


    return render(request, "home.html", {'result': result})

def finding_type(x):
    # New_Visitor	Other	Returning_Visitor
    if x == 1:
        return [1,0,0]
    elif x == 2:
        return [0,0,1]
    else:
        return [0,1,0]
