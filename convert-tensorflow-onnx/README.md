# Convert Tensorflow1 Onnx

<h2>تبدیل مدل تنسورفلو به onnx</h2>

بسته به فرمت مدل ذخیره شده ی تنسورفلو سه حالت  تبدیل مدل تنسورفلو به ONNX داریم:

1- اگر فرمت مدل ذخیره شده بصورت  saved model باشه<br>
2- اگر checkpointهای ذخیره شده رو داشته باشیم<br>
3- اگر graphdef مدل رو داشته باشیم<br>

بسته به هر یک از حالت های ذخیره شده فوق دستور تبدیل مدل tf2onnx فرق میکنه

## References

[ONNX Tutorials](https://github.com/onnx/tutorials#converting-to-onnx-format)

[tensorflow-onnx](https://github.com/onnx/tensorflow-onnx)

## Environment Setup

 <h5>نصب کتابخانه های مورد نیا ز: </h5>

```
pip install -r requirements.txt
```

## How to use


 <h5>تبدیل مدل tf به onnx : </h5>

<h5>1- اگر فرمت مدل ذخیره شده بصورت saved model باشه  </h5>


```
python -m tf2onnx.convert --saved-model tensorflow-model-path --output model.onnx

```

 در کامند فوق بصورت پیش فرض از opset=13  برای تبدیل مدل استفاده شده. اگر نیاز هست که مدل از ورژن های دیگر opset استفاده کنه می تونیم آرگومان `opset--` رو به کامند فوق اضافه کنیم. (برای تبدیل برخی از مدل ها که به ارور میخوریم هم میتونیم  opset رو تغییر بدیم و با مقادیر مختلف opset تست کنیم)
<br>مثلا:

```
python -m tf2onnx.convert --saved-model tensorflow-model-path --opset 16 --output model.onnx

```

<h5>2- اگر checkpointهای ذخیره شده رو داشته باشیم  </h5>

 اگر فرمت مدل تنسورفلو غیر از حالت اول یعنی saved model باشه نیاز هست که نام های ورودی و خروجی مدل رو مشخص کنیم.  برای این منظور برنامه [netron](https://netron.app/) رو نصب میکنیم و مدل رو با این بر نامه باز میکنیم و اسم ورودی و خروجی هارو میتونیم ببینیم.

```
python -m tf2onnx.convert --checkpoint tensorflow-model-meta-file-path --output model.onnx --inputs input0:0,input1:0 --outputs output0:0
```


<h5>3- اگر graphdef مدل رو داشته باشیم  </h5>

در این حالت هم مانند حالت دوم نیاز هست که نام های ورودی و خروجی مدل رو مشخص کنیم که میتونیم از همون برنامه netron استفاده کنیم.

```
python -m tf2onnx.convert --graphdef tensorflow-model-graphdef-file --output model.onnx --inputs input0:0,input1:0 --outputs output0:0
```

مثال استفاده از برنامه netron و مشخص کردن نام های ورودی و خروجی مدل برای graphdef :

نام های ورودی:
![input_names](https://user-images.githubusercontent.com/82975802/226713002-7ab09f0b-4273-42cc-994a-62099b718f46.png)


نام های خروجی:
![output_names](https://user-images.githubusercontent.com/82975802/226713042-64143543-3165-42c8-9eaa-23b2eef282b4.png)

کامند تبدیل مدل تنسورفلو به onnx برای نام های ورودی و خروجی مثال فوق:

```
python -m tf2onnx.convert --graphdef mars-small128.pb --output model.onnx --inputs images:0 --outputs features:0
```


