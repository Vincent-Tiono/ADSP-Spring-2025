import sounddevice as sd
import soundfile as sf
import numpy as np

def sound(y, Fs=8192):
    """
    Plays the audio signal y at the sample rate Fs.

    Args:
        y (array_like): Audio data.
        Fs (int, optional): Sample rate in Hz. Defaults to 8192.
    """
    y = np.asarray(y)
    sd.play(y, samplerate=Fs)
    sd.wait() # Wait until playing is finished

def get_pitch(note: str, base_freq: float ):
    note_map = {
        1: base_freq,  
        2: base_freq * (2**(2/12)),  
        3: base_freq * (2**(4/12)),  
        4: base_freq * (2**(5/12)),  
        5: base_freq * (2**(7/12)),  
        6: base_freq * (2**(9/12)),  
        7: base_freq * (2**(11/12)),  
    }
    symbol_map = {
        "#": 2**(1/12),  
        "b": 2**(-1/12),  
        "^": 2,  
        "v": 0.5,  
    }
    if len(note) == 1: return note_map[int(note)]
    elif len(note) == 2:
        if note[0] != '#' and note[0] != 'b':
            if note[1] == "^":
                return note_map[int(note[0])] * 2
            elif note[1] == "v":
                return note_map[int(note[0])] / 2 
        else:
            if note[0] == "#":
                return note_map[int(note[1])] * (2**(1/12))
            elif note[0] == 'b':
                return note_map[int(note[1])] * (2**(-1/12))
    elif len(note) == 3:
        return note_map[int(note[1])] * symbol_map[note[2]] * symbol_map[note[0]]
    else:
        raise ValueError("Invalid note format. Use '1' to '7' or '#1', 'b2', etc.")

def getmusic(score: list, beat: list, name, bpm, base_freq=261.63, amp=1):
    """
    Args:
    score (list): list of numbered musical notation
    beat (list): list of beat
    name (str): name of music
    bpm (int) : beat per minute
    base_freq (float): Base frequency of the note, default is 261.63 Hz (C4)
    amp (float): Amplitude of sine waves , 0.1 <= amp <= 5

    Additional features:
    1. adjustable amplitude
    2. harmonics for each note
    3. adjustable tempo (bpm)
    4. play music with different key signatures (i.e different base frequencies)
    5. sharp or flat for notes, e.g. '1#' or '2b'
    6. 1 octave up or down, e.g. '1^' means note 1 with 1 octave up, '2v' means note 2 with 1 octave down
    7. decay envelope for each note, so that the consecutive notes with same pitch sound separated.
       (comment out the decay envelope to hear the difference in "paganini_24" case)

    note: feature 5 and 6 can be used together, e.g. '#1^' means note 1 with 1 octave up and sharp
    
    """
    if amp > 5:
        print("Amplitude too high, setting to 5")
        amp = 5
    elif amp < 0.1:
        print("Amplitude too low, setting to 0.1")
        amp = 0.1
    Fs = 11025  # Sample rate
    base_duration = 60/bpm # duration of each beat in seconds
    music = []
    for i in range(len(score)):
        duration = base_duration * beat[i]  # Duration of the note in seconds
        note_freq = get_pitch(score[i], base_freq)  # Frequency of the note
        t = np.linspace(0, duration, int(Fs * duration), endpoint=False)
        note = amp * np.cos(2 * np.pi * note_freq* t)  + amp * 0.3 *np.cos(2*np.pi * 2*note_freq* t) + amp * 0.05 * np.cos(2*np.pi*4*note_freq*t) # add harmonics

        # decay envelope ===================
        decay_duration = 0.2 * duration
        decay_start_index = int(Fs * (duration - decay_duration))
        decay_end_index = int(Fs * duration)
        decay_values = np.linspace(1.0, 0.0, decay_end_index - decay_start_index)
        note[decay_start_index:decay_end_index] *= decay_values # apply decay
        # =================================

        note /= np.max(np.abs(note))  # Normalize to avoid clipping
        music.append(note)
    music= np.concaten(music)
    print(f"Generated {name} with BPM: {bpm}")
    # sound(music, Fs)  # Play the sound
    sf.write(f'{name}.wav', music, Fs)

# score = ['1', '1', '5', '5', '6', '6', '5']
# beat = [1, 1, 1, 1, 1, 1, 1]
# name = "twinkle"

# =========================================
# score = ['3', '2', '3', '6', '3', '2', '3', '7', '3', '2', '3', '1^', '7', '5',
#          '2', '3', '6v', '2', '3', '6v', '5v', '6v']
# beat = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
#         1, 1, 2, 1, 1, 1, 1, 2]
# name = "big_fish"
# ================================
score1 = ['6', '6', '6', '1^', '7', '6', '3^', '3', '3', '#5', '#4', '3', 
         '6', '6', '6', '1^', '7', '6', '3^', '3']
beat1 = [0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 
        0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 1, 1]
score2 = ['6^', '6^', '6^', 'b7^', '6^', '5^',   '4^', '2^', '2^', '4^', '3^', '2^',  '5^', '5^', '5^', '6^', '5^', '4^',   '3^', '1^', '1^', '3^', '2^', '1^',
          '4^', '7', '7', '2^', '1^', '7' ,   '3^', '6', '6', '1^', '7' ,'6',   '4', '#2^', '3', '3^', '2^', '7', '6', '6v']
beat2 = [0.75, 0.25, 0.25, 0.25, 0.25, 0.25,   0.75, 0.25, 0.25, 0.25, 0.25, 0.25,  0.75, 0.25, 0.25, 0.25, 0.25, 0.25,  0.75, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.75, 0.25, 0.25, 0.25, 0.25, 0.25,   0.75, 0.25, 0.25, 0.25, 0.25, 0.25,   0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 1, 0.5] 
score = score1 + score1 + score2
beat = beat1 + beat1 + beat2
name = "paganini_24"


getmusic(score, beat, name, 120, 261.63 , amp=0.1)