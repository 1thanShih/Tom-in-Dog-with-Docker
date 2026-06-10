float32 offset
float32 angle
# 前瞻 anchor（看得比較前方、ANCHOR_RATIO 較小）的量測，供轉彎後對正（T3_ALIGN）使用。
# 一般循線仍用上面 offset/angle（近處 anchor）。偵測不到時為 0.0，與 offset/angle 慣例一致。
float32 offset_far
float32 angle_far
