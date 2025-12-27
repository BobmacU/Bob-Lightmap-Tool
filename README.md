# 💡 Bob's Lightmap Tool
> 🛠️ **A tool for cropping Unity atlas lightmaps and applying them over diffuse with a lot of options.**  
> I used this tool to port [**this map!**](https://steamcommunity.com/sharedfiles/filedetails/?id=3631141109) 

---
<img width="851" height="842" alt="Screenshot 2025-12-27 154157" src="https://github.com/user-attachments/assets/696f24f5-fe04-4299-bbae-ed7d9378997b" />
<img width="849" height="837" alt="Screenshot 2025-12-27 154152" src="https://github.com/user-attachments/assets/cecf8842-fde1-418d-9f39-48ff61b7612d" />

## 🎯 What Is This?
This is a Python tool for helping with getting Unity's lightmap to work and apply the diffuse textures to lightmapped textures. 
This is not only for Garry's Mod. But, Can be used everywhere you need. The tool is not perfect but..
I'd be very appreciate if someone who know the Unity lightmap pipeline could correct some stuff as this tool is not **perfect**. So, feel free to fork it and fix it or make your own tool.

---

## ✨ Features
```yaml
⭐ Usable UI : A good UI interface tool for using. No commandline usage.
🔪 Cropping : Just like how Unity's Tiling & Offset works. You can import your lightmap atlas, type in the Tile X,Y and Offset X,Y to crop the object's used lightmap area. Which you can use to apply over diffuse.
💼 PNG, TGA, JPG and BMP support : Supports 4 formats to use. Unity's lightmap are baked as ".HDR/EXR". So, You should convert to "TGA" or "PNG" for lossless in some tools. Don't use XNConvert for converting HDR/EXR.
🗒️ Console log : For logging. Nothing special but useful :)
➕ Modes : For applying lightmaps over diffuse which are "Multiply, Add and Mix".
🖼️ Tonemappers : 2 tonemapping methods for color correction which are "ACES & Reinhard" with adjustable options. You can also select "None" if you don't need it.
⏰ GPU-Accleration : Only if you have Pytorch. I don't even know if it's working correctly or not.
📷 Preview : A function for previewing the result before applying. Only for lightmap applying.
```

## 🔒 Requirements

Pillow 9.0.0
numpy 1.23.0
imageio 2.25.0
Pytorch **(For GPU-Acceleration but optional)**

**Run these in terminal to install them.**
```yaml
pip install Pillow==9.0.0 numpy==1.23.0 imageio==2.25.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
**For PyTorch**
You should install the one that matches your GPU's CUDA version.
Run **nvidia-smi** in the terminal and check it.

## 🔨 Usage
**For cropping lightmaps**

You should copy the "HDR" files of the lightmap atlas or the lightmap file. Convert it to "TGA" or 'PNG". Import it and then enter "Tile X,Y and Offset X,Y". 
which can be found in the inspector when a lightmapped object is selected. Just make sure the numbers are correct as sometimes, It's just annoying to type a lot of numbers.
After that, Click "Selct Output Folder". Set your export path and click "Crop Lightmap'. It'll export the cropped lightmap.

**For applying lightmaps**

Import the diffuse texture and then cropped lightmap. Select the mode. "Multiply" is the best and the correct one. I don't even know why I added others. You can use tonemapping or leave it.
Adjust any options you want to and then, press "Preview" for a preview of your final result. When satisfied. Click "Selct Output Folder". Set your export path and then press "Apply Lightmap". It'll export into the last folder where you imported lightmap/diffuse if you don't select any export path.

That's it. It's simple tool. But, as I've mentioned above. **Feel free to fork this tool. Make your own tool or fix this tool.** I'll be using it in the next project.

## Known issues & wanted features.

- Sometimes, the result won't be accurate to the Unity. Depending on color space or something that I'm not sure as I'm not a Unity user. But, You can tweak tonemapping to match it much as you can.
- GPU acceleration might have the same speed as CPU acceleration.
- Applied results sometimes get artifacts. I fixed it as I can but still. some artifacts.
- HDR & EXR support needed!

**65% of the tool is written by a friend and Copilot..**
