# Fastest Precise Configuration

This configuration uses all the precise optimizations:

* Bilateral Filter (HW) :  BF_Pipe, BF_Unroll, BF_Pad
* Tracking (HW)			    :  Tr_Pipe
* Integration (HW)      :  Int_Pipe, Int_Unroll, Int_Inter
* Raycast (SW)          :  baseline unoptimized implementation
