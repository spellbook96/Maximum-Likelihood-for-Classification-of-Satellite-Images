from PIL import Image
import numpy as np

img = Image.open('Irabu.bmp')  #画像読み込み
rgb_im = img.convert('RGB')
width, height = img.size
output = Image.new('RGB', (width,height)) #出力用ファイル
img_pixels = np.array([[rgb_im.getpixel((x,y)) for x in range(width)] for y in range(height)])

pos1 = np.array([[24,71],[91,81],[106,134],[52,159],[255,81],[221,138],[45,277],[227,90],[32,17],[144,22]])
pos2 = np.array([[257,322],[207,267],[238,244],[287,136],[147,332],[208,344],[226,302],[157,284],[225,240],[227,330]])
pos3 = np.array([[314,272],[336,287],[293,305],[326,336],[352,304],[392,128],[283,306],[394,124],[323,278],[244,298]])
pos4 = np.array([[246,336],[257,317],[359,119],[309,221],[333,149],[342,120],[344,238],[285,269],[258,321],[309,220]])
pos5 = np.array([[299,176],[302,160],[299,183],[312,170],[311,148],[314,141],[315,133],[282,182],[301,166],[298,186]])
training_data_postion = np.array([pos1,pos2,pos3,pos4,pos5])

training_data = np.zeros([5,10,3])  #教師用データ（r,g,b）
mean_vector = np.zeros([5,3])  #平均ベクトル
covariance_matrix = np.zeros([3,3])  
covariance_matrix_arrange = np.zeros([5,3,3])  #各クラス共分散行列
res_class = np.zeros([height,width]) #各画素がどのクラスに入るかを格納
pxk = np.zeros(5)
for i in range(5): #教師データ(r,g,b)を格納
    for j in range(10):
        training_data[i][j]= rgb_im.getpixel((int(training_data_postion[i][j][0]),int(training_data_postion[i][j][1])))

for i in range (5): #平均ベクトルを算出
    mean_vector[i]=  np.mean(training_data[i],axis=0)
i=0
for data in training_data: #共分散行列を計算
    covariance_matrix = np.cov(data,rowvar=False)
    for j in range(3):
        for k in range(3):
            covariance_matrix_arrange[i][j][k]=covariance_matrix[j][k]
    i+=1

def mnd(_x, _mu, _sig):  #正規分布の計算
    x = np.matrix(_x)
    mu = np.matrix(_mu)
    sig = np.matrix(_sig)
    a = np.sqrt(np.linalg.det(sig)*(2*np.pi)**sig.ndim)
    b = np.linalg.det(-0.5*(x-mu)*sig.I*(x-mu).T)
    return np.exp(b)/a

for i in range(height): #各ピクセルのpxkを計算し、最も大きいpxkによってクラスタリング
    for j in range(width):
        for k in range(5):
            pxk[k] = mnd(rgb_im.getpixel((j,i)),mean_vector[k],covariance_matrix_arrange[k])
        c = np.argmax(pxk)
        output.putpixel((j,i), (int(mean_vector[c][0]), int(mean_vector[c][1]), int(mean_vector[c][2]))) 
output.save('output.bmp')