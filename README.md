# Conf2 

This configuration uses the 15 most impactful optimizations:

* Bilateral Filter (HW) :  BF_Pipe, BF_Unroll, BF_Coeff
* Tracking (HW)			    :  Tr_Pipe, Tr_LP, Tr_LvlIter
* Integration (HW)      :  Int_Pipe, Int_Unroll, Int_Inter, Int_SLP
* Raycast (SW)          :  R_Step, R_LP, R_TrInt, R_Fast, R_Rate
