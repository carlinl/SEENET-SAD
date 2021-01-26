# SEENet-SAD Pytorch
Original enet-sad link is attached below, and codes in this repo is built all based on the ENet-SAD.
* [ENet-SAD](https://github.com/InhwanBae/ENet-SAD_Pytorch)

My work is only add SELayer to the basic block, and it can make the lane lines more contigious, it only costs 0.003s more than original ENet-SAD, which is test on 1060Ti. Two test results videos are output_cv.mp4(ENet-SAD) and output_cv11.mp4(SEENet-SAD), hope you can enjoy!
