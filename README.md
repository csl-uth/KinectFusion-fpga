# Fastest Approximate Configuration

Our fastest configuration uses for each kernel the following optimizations:

* Bilateral Filter (HW) :  BF_Pipe, BF_Unroll, BF_Pad, BF_Coeff
* Tracking (HW)			:  Tr_Pipe, Tr_LP, Tr_LvlIter
* Integration (HW)      :  Int_Pipe, Int_Unroll, Int_Inter, Int_NCU, Int_SLP, Int_HP, Int_Br, Int_FPOp
* Raycast (SW)          :  R_Step, R_LP, R_TrInt, R_Fast, R_Rate
