import os
import pandas as pd
import numpy as np

# مجلد ملفات الـ TXT
folder_path = r"..\data\raw\daphnet"

# قائمة لتجميع كل البيانات
all_data = []

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        
        # قراءة الملف بدون أسماء أعمدة
        df = pd.read_csv(file_path, sep=r"\s+", header=None)
        
        # اختيار الأعمدة المطلوبة بالترتيب: time , X , Y , Z , label
        # لاحظ أنك أردت تبديل Y و Z، لذا وضعنا الأعمدة هكذا: 0,1,3,2,10
        df = df[[0, 1, 3, 2, 10]]
        
        # إعادة تسمية الأعمدة
        df.columns = ["time", "Acc_X", "Acc_Y", "Acc_Z", "label"]
        
        # إضافة اسم الملف لمعرفة Session
        df["file"] = filename
        
        # إضافة الداتا للقائمة
        all_data.append(df)

# دمج كل الملفات في DataFrame واحد
final_df = pd.concat(all_data, ignore_index=True)

# 1) حفظ بصيغة CSV
final_df.to_csv("..\data\filtered\Daphnet_ankle_acc.csv", index=False, encoding="utf-8")

# 2) تحويل البيانات إلى numpy ثم حفظها بصيغة NPZ
data_np = final_df.to_numpy()
np.savez("..\data\filtered\Daphnet_ankle_acc.npz", data=data_np)

print(" تم إنشاء ملفات CSV و NPZ بنجاح!")
