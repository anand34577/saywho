# First steps

Open saywho in your browser: <http://127.0.0.1:7860> on the computer where it runs (or the address of your server).

## 1. Wait for the model

The top right of the page shows what saywho runs on. Right after starting it says **Loading model...** for up to a
minute. When it shows your processor or graphics card (for example `CPU (6 threads, PyTorch)` or
`NVIDIA GeForce RTX 3060 (CUDA)`), it's ready.

## 2. Try the sample

Click **Try the sample**. saywho analyses a 74-second scene from *The Importance of Being Earnest*, read by
LibriVox volunteers (it's in the public domain). After a second or two you should see **3 speakers detected**: two
actors and the narrator reading the stage directions.

Press play on the audio player. The **Speaking now** label follows along, and the matching row in the table lights
up.

## 3. Use your own recording

Drag a file onto the big box, or click it to pick one. Almost any audio or video file works: phone voice memos,
Zoom or Teams recordings, podcasts, MP4 videos and so on. Video files are fine; saywho only looks at the sound.

How long it takes depends on your hardware: on a normal laptop an hour of audio takes a couple of minutes, with an
NVIDIA graphics card well under a minute. Uploading a large video over a slow network can take longer than the
analysis itself.

Your file is analysed on the computer where saywho runs and deleted straight after. Nothing is sent to the internet.

## 4. Name the speakers

saywho calls people Speaker 1, Speaker 2 and so on, in the order they first speak. Click a name in the
**Speakers** section and type a real one, for example "Priya". The timeline, the table and the downloads all use
the new names.

## 5. Download the result

At the top of the **Segments** table:

| Button | You get | Good for |
|---|---|---|
| **SRT** | a subtitle file | load it in VLC, YouTube or a video editor next to the video, to see who is talking |
| **CSV** | a spreadsheet | open in Excel, Google Sheets or LibreOffice |
| **JSON** | structured data | your own scripts |
| **RTTM** | the research format | comparing against other diarization tools |

## 6. If the result looks off

The three boxes above the results change how the model's output is turned into segments. Changes apply instantly,
without running the model again:

- **Speech threshold** (default 0.5): how sure the model must be that someone is speaking. Lower it (0.3 or 0.4)
  if quiet speech is missed; raise it (0.6 or 0.7) if noise shows up as speech.
- **Min segment** (default 0.1 s): drops anything shorter than this, like coughs and "mm-hm". Raise it to 0.5 for a
  cleaner timeline.
- **Merge gaps under** (default 0.3 s): joins two pieces of the same person's speech if the pause between them is
  shorter than this. Raise it to 1 if one sentence keeps getting split up.

If two different people get the same label, or one person is split into two speakers, that's the model itself.
Very similar voices, poor microphones and more than eight speakers make it harder.
[Reading the results](Reading-the-results) explains every number on the page.
