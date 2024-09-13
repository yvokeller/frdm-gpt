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

## Fine-tuning Llama-3.1-8B (a year later)

I came back to project and tried to fine-tune the Llama-3.1-8B model to write prose in my style from given scene beats. Used the mlx framework by Apple which made it super easy to quantize and fine-tune the model using QLoRA on my M2 Max MacBook Pro. A first attempt failed pretty miserably, I overfitted the model in just one epoch - likely the learning rate was too high.

I created the dataset using GPT-4o-mini to generate scene beats for each chapter of my novel. I then turned things around and fine-tuned the Llama-3.1-8B model such that it learns to write prose in my style from given scene beats. (see `dataset-scene-beat-to-prose.ipynb`)

Here is an example of a scene generated from a set of scene beats:

Scene Beats:

- Raven kniet verzweifelt im Schlamm und versucht, Caleor aus seiner Ohnmacht zu holen. Die Angst um Cale und die bevorstehende Gefahr treibt ihn an, während er gleichzeitig von der drängenden Realität überwältigt ist.
- Caleor erwacht, panisch und desorientiert, konfrontiert mit der erschreckenden Wahrheit, dass sie sich in einer gefährlichen Lage befinden. Seine Fragen über den Zustand der Dinge spiegeln seine innere Unruhe wider.
- Der Protagonist, Dusk, analysiert die Situation und versucht, Caleor zu beruhigen, während er die Dringlichkeit der Flucht vor den Werwölfen betont.
- Caleor wird von der schockierenden Wahrheit über das Werwolf-Gift überwältigt, was zu einem emotionalen Konflikt zwischen Wut und Angst führt. Dies drängt ihn, die Verantwortung für die Situation zu übernehmen.
- Die Gruppe erkennt die Notwendigkeit von Caleors Unterstützung, um die Mission zu erfüllen. Dusk fühlt sich erleichtert, dass Caleor bereit ist, ihnen zu helfen, und stärkt den Zusammenhalt unter den Freunden.
- Sie positionieren sich strategisch am Waldfriedhof und bereiten sich auf die bevorstehende Konfrontation mit den Werwölfen vor, während der Regen eine schützende Ablenkung bietet.
- Plötzlich öffnet sich das Portal, und die erste Welle der Werwölfe tritt hervor, was die unmittelbare Bedrohung spürbar macht. Die Situation wird angespannt, als sie erkennen, dass sie schnell handeln müssen.
- Dusk beobachtet die Ankunft der Werwölfe und stellt fest, dass sie Schwierigkeiten haben werden, ihren Plan ohne mehr Informationen über Versiporcus umzusetzen.
- Trotz der gefühlten Ausweglosigkeit motiviert Caleor die Gruppe, durchzuhalten, und sie setzen den Plan in Bewegung, um die Werwölfe heimlich zu verfolgen.
- Raven übernimmt die Ablenkung, während Dusk und Caleor das Gewehr laden, was einen letzten verzweifelten Versuch darstellt, die Situation zu kontrollieren.
- Die Spannung steigt, als sie die Werwölfe verfolgen, und Dusk ist bereit, seine Schüsse abzugeben, erkennt jedoch, dass die Gelegenheit zu schlüpfrig ist, um richtig zu handeln.
- Die Konfrontation erreicht ihren Höhepunkt, als Versiporcus und Custos plötzlich auftauchen, was Caleors Panikattacke auslöst und das Überleben der Gruppe auf die Probe stellt.

Generated Scene (Original Llama-3.1-8B)

```text
Raven kniete im Schlamm, sein Gesicht verzerrt vor Sorge, als er Caleor aus seiner Ohnmacht zu holen versuchte. Seine Hände tasteten über den Körper seines Freundes, in der Hoffnung, ihn zurück ins Bewusstsein zu bringen. Der Regen prasselte um sie herum, und die Dunkelheit um sie herum schien zu leben. Raven wusste, dass sie in einer gefährlichen Lage waren, und die Angst um Caleor und die bevorstehende Gefahr trieb ihn an, wobei er gleichzeitig von der drückenden Realität überwältigt war.
Caleor erwachte plötzlich, panisch und desorientiert. Sein Blick suchte die Umgebung ab, und als er Raven sah, erkannte er die erschreckende Wahrheit, dass sie sich in einer gefährlichen Lage befanden. Seine Fragen über den Zustand der Dinge spiegelten seine innere Unruhe wider. 'Was...was ist passiert? Wo sind wir?'
Der Protagonist, Dusk, analysierte die Situation und versuchte, Caleor zu beruhigen, während er die Dringlichkeit der Flucht vor den Werwölfen betonte. 'Caleor, wir müssen hier raus, jetzt. Wir müssen weg von hier, bevor die Werwölfe kommen.' Caleor schüttelte den Kopf, sein Gesicht verzerrt vor Schmerz. 'Ich...ich kann nicht...Ich fühle mich so...'
Caleor wurde von der schockierenden Wahrheit über das Werwolf-Gift überwältigt, was zu einem emotionalen Konflikt zwischen Wut und Angst führte. Dies drängte ihn, die Verantwortung für die Situation zu übernehmen. 'Ich...ich bin schuld, dass...dass wir hier sind. Ich hätte es wissen müssen.'
Die Gruppe erkannte die Notwendigkeit von Caleors Unterstützung, um die Mission zu erfüllen. Dusk fühlte sich erleichtert, dass Caleor bereit war, ihnen zu helfen, und stärkte den Zusammenhalt unter den Freunden. 'Caleor, wir brauchen dich. Wir können das nicht ohne dich schaffen.'
Sie positionierten sich strategisch am Waldfriedhof und bereiteten sich auf die bevorstehende Konfrontation mit den Werwölfen vor, während der Regen eine abschreckende Ablenkung bot. Die Stille zwischen ihnen war nur unterbrochen vom Rauschen des Regens und dem Knarren der Bäume.
Plötzlich öffnete sich das Portal, und die erste Welle der Werwölfe trat hervor, was die unmittelbare Bedrohung sichtbar machte. Die Situation wurde angespannt, als sie erkennen, dass sie schnell handeln mussten.
Dusk beobachtete die Ankunft der Werwölfe und stellte fest, dass sie Schwierigkeiten haben würden, ihren Plan ohne mehr Informationen über Versiporcus umzusetzen. 'Wir müssen schnell handeln, aber ich habe keine Ahnung, wo wir uns genau befinden.'
Trotz der gefährdeten Ausweglosigkeit motivierte Caleor die Gruppe, durchzuhalten, und sie setzten den Plan in Bewegung, um die Werwölfe heimlich zu verfolgen. Raven übernahm die Ablenkung, während Dusk und Caleor das Gewehr luden, was einen letzten verzweifelten Versuch darstellte, die Situation zu kontrollieren.
Die Spannung stieg, als sie die Werwölfe verfolgten, und Dusk war bereit, seine Schüsse abzugeben, erkannte jedoch, dass die Gelegenheit zu schnell war, um richtig zu handeln. 'Wir müssen uns beeilen, aber ich habe keine Ahnung, was wir tun.'
Die Konfrontation erreichte ihren Höhepunkt, als Versiporcus und Custos plötzlich auftauchten, was Caleors Panikattacke auslöst und das Überleben der Gruppe auf die Probe stellte.
```

Generated Scene (Fine-tuned Llama-3.1-8B-Write-Beat-to-Prose-v2)

```text
Raven kniete verzweifelt im Schlamm, seine Hände um Caleors Kopf geschlossen, als er versuchte, ihn aus seiner Ohnmacht zu holen. Die Angst um Cale und die bevorstehende Gefahr treibt ihn an, während er gleichzeitig von der drängenden Realität überwältigt ist. Sein Gesicht war verzerrt vor Sorge, als er Caleors Namen rief, aber der junge Mann antwortete nicht. Raven hob Caleors Kopf an und rieb ihn sanft, um ihn wieder zu Bewusstsein zu bringen. "Cale, hörst du mich? Cale, bitte, wach auf!" Er bettete Caleors Kopf auf seine Schulter und hielt ihn fest, während er ihn sanft schüttelte. Caleor erwachte mit einem Ruck, panisch und desorientiert, und riss die Augen auf. "Was...was ist los? Wo sind wir?" Er sah sich um und sah den Schlamm und die umliegenden Bäume. Sein Blick fiel auf Raven, der ihn besorgt ansah. "Cale, wir müssen hier weg. Die Werwölfe...sie kommen", sagte Raven, während er Caleor half, sich aufzurichten. Caleor sah ihn verwirrt an, aber dann erinnerte er sich an die Ereignisse der letzten Nacht. "Oh nein...das ist nicht möglich. Ich bin...ich bin giftig", flüsterte er, als er sich an das Werwolf-Gift erinnerte, das er bei der Begegnung mit dem Werwolf eingefangen hatte. Seine Augen weiteten sich, als die Wahrheit ihn überwältigte. "Nein, nein, nein...was habe ich getan?" Caleor stand auf, und Raven hielt ihn fest, um ihn zu beruhigen. "Cale, es ist okay. Wir sind hier, um dir zu helfen. 
Wir müssen weg von hier", sagte Dusk, der plötzlich neben ihnen stand. "Wir müssen das Gift loswerden. Wir können nicht riskieren, dass es dich kontrolliert", fügte er hinzu. Caleor sah ihn an, und für einen Moment sah man die Angst und die Wut in seinen Augen. "Ich...ich kann nicht einfach weglaufen. Ich muss...ich muss etwas tun", sagte er, während er versuchte, seine Verantwortung zu übernehmen. Raven nickte. "Ja, Cale. Wir brauchen deine Hilfe, um die Mission zu erfüllen. Wir können nicht ohne dich durchkommen." Caleor atmete tief durch und nickte. "Ich bin bereit. Lass uns gehen."
Die Gruppe positionierte sich strategisch am Waldfriedhof und bereitete sich auf die bevorstehende Konfrontation mit den Werwölfen vor. Der Regen, der den Boden durchnässte, half, die Geräusche zu dämpfen, und die Gruppe hoffte, dass es eine schützende Ablenkung bieten würde. Plötzlich öffnete sich das Portal, und die erste Welle der Werwölfe trat hervor, was die unmittelbare Bedrohung spürbar machte. Die Situation wurde angespannt, als sie erkennen, dass sie schnell handeln mussten. 
Dusk beobachtete die Ankunft der Werwölfe und stellte fest, dass sie Schwierigkeiten haben würden, ihren Plan ohne mehr Informationen über Versiporcus umzusetzen. "Das ist nicht gut", flüsterte er, während er Caleor und Raven ansah. Caleor nickte, und seine Augen flammten auf, als er sich an die Situation erinnerte. "Wir müssen durchhalten. Wir können nicht aufgeben", sagte er, während er die Gruppe motivierte. "Okay, okay. Wir tun es. Wir verfolgen sie heimlich und versuchen, mehr über Versiporcus zu erfahren", sagte Dusk, während er das Gewehr lud. 
Raven übernahm die Ablenkung, während Dusk und Caleor das Gewehr luden, was ein letzter verzweifelter Versuch darstellte, die Situation zu kontrollieren. Die Spannung stieg, als sie die Werwölfe verfolgten, und Dusk war bereit, seine Schüsse abzugeben, erkannte jedoch, dass die Gelegenheit zu schnell war, um richtig zu handeln. "Nein, warten Sie", flüsterte er, während er sich umdrehte. "Versiporcus und Custos...sie kommen", sagte er, während die Konfrontation ihren Höhepunkt erreichte. Caleor sah ihn an, und seine Augen weiteten sich vor Panik. "Nein, nein, nein", flüsterte er, während die Gruppe auf die beiden Männer zusteuerte.
```

The fine-tuning defintely makes a difference! However, there is still a lot of room for improvement. The results are not yet something worth reading. I let ChatGPT summarize the good & bad, and agree with most of it. So one would think an iterative approach with a stronger model might eventually yield something interesting.

**Positives of fine-tuned over original model:**

1. **Emotional Depth**: Stronger portrayal of Raven’s care for Caleor, creating a more intimate connection.
2. **Clearer Flow**: Smoother progression of events, making the scene easier to follow.
3. **Effective Dialogue**: Direct and urgent dialogue heightens the tension.
4. **Clear Motivation**: Characters’ actions and stakes are better defined, especially Caleor’s internal struggle.

**Negatives of generated prose:**

1. **Repetition**: Certain phrases and emotions are repeated, weakening the impact.
2. **Telling Instead of Showing**: Emotional states are sometimes stated outright rather than shown through action.
3. **Pacing Issues**: Emotional resolutions happen too quickly, reducing tension.
4. **Flat Descriptions**: The setting could be more vivid and tied into the mood.
5. **Shallow Character Reactions**: Caleor’s emotional shift lacks depth and realism.

I'm excited and optimistic about the potential of this in the future, but there surely are quite some problems to solve before a model that I can run locally will actually be able to support me in writing a full sequel to ANSTURM. Back to work :)
