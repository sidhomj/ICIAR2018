from Use_Net import Use_Net
import sys
imgdir=sys.argv[1]

UNet=Use_Net()
UNet.Get_Data(imgdir)
UNet.Predict()
UNet.df1.to_csv('Prediction_Results.csv',index=False)
