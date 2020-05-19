import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


paths = {
"images_path" : "./JPEGImages",
"targets_path" : "./improved_contours",
"train_names_path" : "./train.txt",
"val_names_path" : "./dummy_val.txt",
"results_path" : "./results",
"models_path" : "./models",
"CRF_mask_path" :"./SegmentationObjectFilledDenseCRF"
}


targets_names = []

with open(paths["val_names_path"]) as handle:
    for name in handle:
        name = name.split("\n")[0].strip()
        targets_names.append(name)




targets_paths = []
results_paths = []
for name in targets_names:
    targets_path = os.path.join(paths["targets_path"], name + ".png")
    targets_paths.append(targets_path)

    result_path = os.path.join(paths["results_path"] , name + ".npy")
    results_paths.append(result_path)




class Evaluator:
    def __init__(self, target_paths , results_paths):
        self.targets_path = targets_paths
        self.results_path = results_paths
    def make_CM(self, target_image , result_image , thres):
        temp_target_image = np.zeros_like(target_image)
        temp_result_image = np.zeros_like(result_image)
        
        temp_target_image[target_image == 255]  =1

        
        temp_result_image[result_image >= thres] = 1

        
        sm = temp_target_image + temp_result_image
        diff = temp_target_image - temp_result_image
        
        tp = (sm == 2).sum()
        tn = (sm == 0).sum()
        fp = (diff == -1).sum()
        fn = (diff == 1).sum()
        
        return tp,tn,fp,fn
    
    def calc_PR_curve(self, step):
        precision = [0]*step
        recall = [0]*step
        thres_arr = np.linspace(0,1,step)
        for target_path , result_path in zip(self.targets_path,self.results_path):
            target = cv2.imread(target_path,0)
            result = np.load(result_path, allow_pickle=True)
            
        
            for i in range(step):
                thres = thres_arr[i] 
                tp,tn,fp,fn = self.make_CM(target,result,thres)
                if(tp + fp == 0):
                    precision[i] += 1
                else:
                    precision[i] += tp/(tp + fp)
                
                if(tp + fn == 0):
                    recall[i] += 1
                else:
                    recall[i] += tp/(tp + fn)
                    
        precision = [x/len(self.targets_path) for x in precision]
        recall = [x/len(self.targets_path) for x in recall]
        
        return precision , recall, thres_arr
            
        
        


if __name__ == "__main__":
        evl = Evaluator(targets_paths , results_paths)
        P, R, thres = evl.calc_PR_curve(300)
        plt.plot(R,P)
        plt.show()
        
        f_score = -1
        f_thres = -1
        index = 0
        for p,r,t in zip(P,R,thres.tolist()):
            if(p == 0 or r == 0):
                continue
            f1 = 2*p*r/(p + r)

            if(f1 > f_score ):
                f_score = f1
                f_thres = t
            index += 1
        print("max f score = " + str(f_score) + " at threshhold = " + str(f_thres/301))
        
            

    
             
     
