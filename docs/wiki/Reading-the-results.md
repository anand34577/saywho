# Reading the results

Everything on the results page, from top to bottom.

![The results page](https://raw.githubusercontent.com/anand34577/saywho/main/docs/images/results-full.png)

## Player

Plays your file. **Speaking now** shows who is talking at the current moment. Click any row in the table, or any
point on the timeline, to jump there.

## Summary

| Box | Meaning |
|---|---|
| Speakers detected | how many different voices saywho found |
| Duration | length of the recording |
| Speech | share of the recording where at least one person is talking |
| Silence / non-speech | time with no one talking: pauses, music, noise |
| Overlapped speech | time when two or more people talk at once, and its share of all speech |
| Speaker turns | how many times the floor changed hands (a person talking in several pieces counts once) |
| Interruptions | turns that started before the previous speaker had finished |
| Avg gap between turns | the usual pause between one person stopping and the next starting |
| Max simultaneous | the most people talking at the same moment |
| Turns / minute | how lively the conversation is |
| Processing time | how long saywho took, upload not included |
| Real-time factor | how many times faster than real time the analysis ran: 100x means an hour of audio in 36 seconds |

The grey line underneath breaks the time down by step (decoding the file, preparing the audio, the model, building
segments). It's useful when comparing hardware.

## Speaker timeline

One row per speaker; each bar is a stretch of speech. Click to jump to that moment. Scroll the mouse wheel over it
to zoom in on part of a long recording.

## Speaker activity probability

The model's raw output: for each speaker, a line showing how sure it is, moment by moment, that this person is
talking. The dashed line is the **Speech threshold**: anything above it counts as speech. If a quiet speaker's line
keeps peaking just under the dashed line, lower the threshold.

## Speakers

A card per person: total talk time and their share of all speech, share of the whole recording, number of segments,
average and longest segment, when they were first and last heard, how long they talked over someone else, and how
many times they interrupted. Click the name to rename the speaker.

## Talk-time share

The same talk times as bars, so you can see at a glance who did most of the talking.

## Who follows whom

A grid of turn changes. Read a row as "after this person stopped, how many times did each other person speak
next?". In an interview, the interviewer and guest alternate; in a meeting, it shows who responds to whom.

## Segments

Every stretch of speech with its speaker, start, end and length. A coloured dot under **Overlaps** means someone
else was talking at the same time. The buttons above the table download the result; see
[First steps](First-steps#5-download-the-result).
