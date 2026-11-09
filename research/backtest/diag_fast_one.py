"""Call sixyear_fast.process_chunk on one day and compare with the known-good day_book."""
import pandas as pd

import bt
import diff_all as D
import sixyear_fast as F

days = F.all_days()[:10]
day = days[5]
lead = days[:5]
res = F.process_chunk(([day], lead))
print("fast:", res)

import sixyear
from sixyear import files, SPOT, FrameData, expiry
from datetime import datetime

allf = files()[:10]
frames = [D.load_csv(d, p) for d, p in allf[:6]]
book = D.day_book(pd.concat(frames, ignore_index=True), day)
print("reference:", book)
