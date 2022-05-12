# NST

Neural Style Transfer is the process of blending a *style* from an image you would like to extract the visual properties from, onto a *content* image while keeping spatial information. Its an application of computer vision using Convolutional Neural Networks. The NST employs a pre-trained Convolutional Neural Network to activate the neurons in a particular way so that the content and style match to preserve content image spatial information while imposing style aspects to the output. 

## Installation and Usage


```
git clone --recursive https://github.com/prayge/NST.git
pip install -r requirements.txt
python main.py --content='png/content.png' --style='png/style.png'
python main.py --contentimage='test/piccy.jpg' --styleimage='test/style.jpg'
```

## Results

The following content image was infered 
