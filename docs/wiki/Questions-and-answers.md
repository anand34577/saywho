# Questions and answers

### Does saywho send my recordings anywhere?

No. Files are analysed on the computer where saywho runs and deleted as soon as the result is ready. The only
internet access is during installation (and on the very first start, to download the model). After that you can
set `HF_HUB_OFFLINE=1` and run it without any connection. The Docker images already include the model.

### Does it write down what people say?

No. saywho finds *who* speaks *when*, not the words. For a transcript, use a speech-to-text tool such as Whisper,
then use saywho's SRT or CSV export to see which speaker said each part.

### Can it tell me who a speaker is?

No, and it isn't meant to. Speakers get numbers in the order they first speak. You can rename them on the page.

### How many speakers can it handle?

Up to eight. With more people in a recording, some will be merged with others.

### What languages does it work with?

saywho works from the sound of the voices, not from the words. The model was built by NVIDIA; its
[model card](https://huggingface.co/nvidia/Nemotron-3-Diarization) describes what it was trained and tested on.
Results are best with clear recordings where people don't talk over each other too much.

### How accurate is it?

Usually good on clear recordings, and worse with background noise, phone-quality audio, similar voices or lots of
people talking at once. Always check the result before you rely on it, especially for anything that affects people.

### Do I need a graphics card?

No. A normal laptop processor analyses an hour of audio in about two minutes. An NVIDIA graphics card makes it a few
times faster, and saywho uses it automatically.

### What files can I use?

Almost any audio or video file: MP3, WAV, M4A, FLAC, OGG, Opus, MP4, MKV, MOV, WebM and more. saywho uses FFmpeg to
read them.

### Is there a size or length limit?

By default, files up to 2 GB and recordings of any length. You can set your own limits in [Settings](Settings).
Long recordings need memory: about 4 MB per minute of audio while it's being analysed.

### Can several people use it at once?

Yes. Requests are handled one after another, so with many users at once each waits a little longer.

### Can I use it in my own program?

Yes. saywho has a REST API: send a file, get the result as JSON or a file. See the
[API reference](https://github.com/anand34577/saywho/blob/main/docs/API.md) and the interactive docs at
`http://localhost:7860/docs`.

### Can I use it commercially?

saywho's code is under the Apache License 2.0, and the model under the OpenMDW License 1.1. Both allow commercial use.
Recordings of people are personal data in many countries, so make sure you're allowed to process them. See
[License and credits](https://github.com/anand34577/saywho#license-and-credits).

### Who makes saywho? Is it an NVIDIA product?

saywho is an independent open-source project. It uses an open model published by NVIDIA, but it isn't made,
endorsed or supported by NVIDIA or Rockchip.
