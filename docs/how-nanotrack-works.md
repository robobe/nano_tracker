# How NanoTrack works

NanoTrack is a **single-object tracker**. You tell it what object to follow once, by drawing a rectangle around it in the first video frame. From then on, it tries to answer one question for every new frame:

> “Where is the thing that looks most like the object I was shown?”

It is not a general object detector. It does not start by finding every person, car, or cat in a frame. It starts with your chosen object and follows that one object.

## The big idea

Imagine you have a small passport photo of a backpack. In the next video frame, you do not search every pixel of the whole world. You look in the area near where the backpack was last seen and ask: “Which part looks most like my passport photo?”

NanoTrack does that with learned number patterns instead of human eyes.

## ONNX, PNNX, and NCNN in one minute

These names describe different stages of getting a neural-network model ready to run:

| Name | Simple meaning | Used here for |
| --- | --- | --- |
| **ONNX** | A common, portable model format—like a standard document format for AI models. | The original NanoTrack V3 `.onnx` files run with ONNX Runtime. |
| **PNNX** | NCNN's conversion tool and its optional intermediate model representation. | It reads the ONNX models and produces NCNN-ready files. It is used during conversion, not while tracking. |
| **NCNN** | A lightweight inference engine and its `.param` + `.bin` model format. | The C++ NCNN tracker loads the converted files and runs them on CPU. |

So the route in this project is:

```text
NanoTrack ONNX model --PNNX conversion--> NCNN .param + .bin model --NCNN runtime--> tracking result
```

PNNX does not retrain NanoTrack or invent a new object tracker. It translates the already-trained model into a form NCNN can load efficiently. The original ONNX files remain useful because other runtimes, such as ONNX Runtime, understand them too.

![Original NanoTrack V1 architecture diagram](https://raw.githubusercontent.com/HonglinChu/SiamTrackers/master/image/nanotrack_network.png)

*The diagram is the upstream NanoTrack V1 architecture image. This project uses NanoTrack V3, but both versions use the same template-plus-search idea. Image source: [HonglinChu/SiamTrackers](https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack), Apache-2.0.*

## One complete tracking step

```text
first frame + your rectangle
        |
        v
template crop -> backbone -> saved template features

next video frame
        |
        v
search crop -> same backbone -> search features
        |
        v
matching head -> 15 x 15 candidate locations
        |
        v
pick the safest best candidate -> draw the new rectangle
```

The template is calculated once when tracking starts. The search path is calculated once for every later frame.

## Step 1: make the template

Your selected rectangle is not used exactly by itself. NanoTrack also includes some space around it. This extra area is called **context**. Context helps because an object often has useful surroundings: a face has hair, a car has road, and a box may have a table behind it.

The project crops that area, resizes it to **127 × 127 pixels**, and calls it the **template**. Think of the template as the tracker’s saved “what I am looking for” card.

The backbone model changes the 127 × 127 colour image into a smaller collection of learned features with shape **96 × 8 × 8**:

- `96` means it has 96 different learned pattern channels.
- `8 × 8` means each channel is an 8-by-8 map, not an ordinary photograph.

One channel may react strongly to an edge, another to a texture, another to a shape part. We do not name these channels by hand; the model learned them during training.

## Step 2: make the search area

For the next frame, NanoTrack assumes the object probably did not teleport across the screen. It crops a larger square around the object’s previous centre, then resizes it to **255 × 255 pixels**. This is the **search** image.

The same backbone reads the search image and creates **96 × 16 × 16** search features. The search map is larger than the template map because the tracker needs room to check several nearby positions.

If the object is moving fast, leaves this search area, or is completely hidden for many frames, the tracker may lose it. That is the cost of being fast: it searches nearby, not everywhere.

## Step 3: the matching part — the important bit

The matching head receives two sets of features:

- template features: “this is what my object looks like”
- search features: “these are the possible places in the new frame”

It does **not** compare raw red, green, and blue pixels. Raw pixels change a lot when lighting, blur, scale, or viewpoint changes. Instead, it compares the backbone’s learned feature patterns.

### A tiny matching example

Pretend one template feature is a short list:

```text
template: [edge=high, blue=medium, round=high]
place A:  [edge=high, blue=medium, round=high]
place B:  [edge=low,  blue=high,   round=low]
```

Place A gets a higher match score because its pattern agrees more with the template. In real NanoTrack, the lists are much larger and the model learns which details matter.

### Pixel-wise correlation

The V3 head first changes both feature sets with small learned layers. It then compares each template position with every search position using dot products (multiplying matching feature values and adding them).

This creates a detailed answer to questions like:

> “Does the top-left part of the saved object resemble this part of the search area?”

NanoTrack V3 uses this **pixel-wise correlation** to keep fine details. A channel-attention step then turns down less useful matching channels and turns up useful ones.

### Depth-wise correlation

V3 also performs **depth-wise correlation**. Instead of mixing all 96 feature channels together immediately, it compares corresponding channels separately. It uses the central `4 × 4` area of the template feature map for this path.

This is like asking 96 specialists for opinions:

- one specialist notices a vertical edge;
- another notices a colour/texture clue;
- another notices a shape clue.

Keeping those opinions separate at first is efficient and preserves useful information. The head joins the pixel-wise and depth-wise results, then reduces them back to 96 channels with a small learned layer.

### Two answers for every possible location

After matching, the V3 head has separate small networks for two jobs. It produces two maps of size **15 × 15**:

1. **Classification map:** two values at every location: “background” and “object”.
2. **Box map:** four values at every location: distance to the object’s left, top, right, and bottom edges.

So NanoTrack does not make only one guess. It makes **225 guesses** (`15 × 15`), scores every guess, and chooses one.

## Step 4: turn the maps into rectangles

Each square in the 15 × 15 map represents a point in the search image. Neighbouring points are 16 pixels apart in model coordinates. The centre point means “near the old location”; points farther away mean “try a farther move.”

For one candidate point `(x, y)`, the four box values are `left`, `top`, `right`, and `bottom`. The code turns them into corners like this:

```text
x1 = x - left       y1 = y - top
x2 = x + right      y2 = y + bottom
```

From those corners it gets the candidate rectangle’s centre, width, and height.

The object score is calculated from the two classification values. In plain language: if the “object” value is much stronger than the “background” value, the score is close to 1; otherwise it is closer to 0.

## Step 5: do not believe every high score

A bright, similar-looking object can sometimes get a high score by accident. NanoTrack adds three safety checks before choosing its final rectangle:

1. **Shape and size penalty:** a candidate is penalized if it suddenly becomes much wider, taller, larger, or smaller than the previous object.
2. **Centre preference:** a Hann window gives a small bonus to candidates near the old centre. This prevents wild jumps when two places look similar.
3. **Slow update:** the final width and height move partway toward the new box instead of instantly changing all the way.

For this V3 project, the final ranking is based on:

```text
rank = penalty × object_score × (1 - 0.455) + centre_bonus × 0.455
```

The selected box then updates the tracker. Its update speed depends on confidence and penalty, with a maximum learning-rate setting of `0.348`. A confident, sensible-looking match can move the box more; a strange-looking match moves it less.

These rules are why the tracker can look calmer than simply choosing the biggest raw score.

## What happens on the next frame?

The new rectangle becomes the centre of the next search area. The original template features stay saved; NanoTrack V3 in this project does not replace them every frame.

Keeping the first template is useful when the object temporarily changes because of blur or partial hiding. But it also has a limitation: if the object truly changes appearance a lot, the saved template can become too old.

## Why tracking can fail

Tracking is a best guess, not a promise. It can fail when:

- the object leaves the local search area;
- another object looks very similar;
- the object is fully hidden for too long;
- the first rectangle includes too much background or misses important parts of the object;
- the object changes shape, lighting, or viewpoint more than the learned features can handle.

When that happens, stop and select the object again. For finding an object anywhere in a frame after it is lost, combine a detector with a tracker; NanoTrack alone is designed for following one already-known object.

## Where this appears in this repository

- `nanotracker.py` contains the Python ONNX Runtime version.
- `cpp/src/ncnn_main.cpp` contains the same crop, matching-output decoding, ranking, and update logic for NCNN CPU inference.
- `models/nanotrackv3/` holds the two V3 ONNX models: backbone and head.
- `models/nanotrackv3_ncnn/` holds their converted NCNN `.param` and `.bin` files.

The V3 values used here—window influence `0.455`, penalty `0.138`, and learning rate `0.348`—match the upstream V3 configuration. See the original [NanoTrack README](https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack) and its [V3 matching-head source](https://github.com/HonglinChu/SiamTrackers/blob/master/NanoTrack/nanotrack/models/head/ban_v3.py) for the original implementation.

[← Return to README.md](../README.md)
