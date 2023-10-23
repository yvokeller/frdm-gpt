# frdm-GPT

Fun sunday project, implementing a "Attention Is All You Need"-style GPT model, after reading the paper for the first time. I based this on Andrej Karpathy's great "Let's build GPT: from scratch, in code, spelled out" lecture available on [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY&t), with a little twist:

I am using the text of my 2017 german fantasy novel "ANSTURM" as the dataset, instead of Shakespeare.

Let's see what the model comes up with. Maybe I can even write the sequel with it's help :)

## Results

I trained a first bigger model with a total of 10'801'233 parameters, and a context window of 256 tokens (character based). Took around 40 minutes on my M2 Max. The results are... interesting. The model seems to have learned some basic german, but it's still far from being able to write a novel on it's own. But it's a start.

Below are two examples:

`python inference.py --prompt 'Sokrate liess den Wald mit den Werwölfen hinter sich und'`

I find it quite fascinating that the model is already picking up central topics from the novel, like "Verwandlung", "Plan", "Portal", knows some character names (Trest), and is even able to write basic dialogue.

> Sokrate liess den Wald mit den Werwölfen hinter sich und > nehme es wieder oder später will kämpfen, die weiteren Plan nicht ausgearbeitet hat, wie alles diese umzukommenen Verwandlung noch auf einem Jahreszeitsnummer 25, wer also nehmen wir alle heran? Hat die Zeit vorher brauchen können? Nein, man streitet sich natürlich nicht entgangen. Das war uns auch heute die Nacht auf Montag vom Plan, persöne die Sache entfernt eine gementliche Wasser mit zu wollen. Sie verrückt einen Namen. So sind davon erzählt darauf, hier alles muss sich wohl hier.«

> Sokrate liess den Wald mit den Werwölfen hinter sich und > nach kommen. Die weißen Glühbirnen, die nächste reißen Tür auf dem Haus dem Kakao gehen. Der Wolf ist mir nicht so stark bemerkt zu haben. Ich richte mich Gutes zu, bis zu Trest aus den Portal gehen habe ich es durch den Kopf. Dabei für ein Verstand gebaut habe. Er ist seinen Neugierig geworden?
»Heute hast du alles nur entdeckt?«
»Nein, man streitet sich bis heute darüber. Manche sagen, er hätte von Kommt. Muss dich zu gewöhnlichen unenthalten. Der Frage ist ein speziellen, was gar nicht los,

## The book

In case you are interested: My novel is available as an e-book on Amazon, Apple Books, Google Play Books, Thalia, Weltbild, Hugendubel, and many more. Just search for ANSTURM by Yvo K.

Or, even better - Get it directly in my shop: [ANSTURM](https://frdmauthor.net/shop/)
