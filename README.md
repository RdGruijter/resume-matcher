Resume & Job Description Matcher (PoC)
Een Proof of Concept (PoC) applicatie gebouwd met Streamlit en SpaCy die werkzoekenden helpt hun CV te optimaliseren voor specifieke vacatures. De tool analyseert de tekst op basis van trefwoorden en semantische gelijkenis om een nauwkeurige "match score" te geven.

🚀 Functionaliteiten
Keyword Extractie: Gebruikt Natural Language Processing (NLP) om automatisch relevante vaardigheden, rollen en kwalificaties te identificeren.

Semantische Matching: De tool kijkt verder dan exacte woorden. Dankzij word embeddings (en_core_web_md) herkent het dat termen als "MSc" en "Master's degree" hetzelfde betekenen.

Sectie-gebaseerde Analyse: De applicatie splitst je CV op in secties (zoals 'Skills', 'Experience', 'Education') om gerichte verbeterpunten te suggereren.

Contextuele Suggesties: Geeft specifiek advies over waar en hoe je ontbrekende keywords kunt toevoegen (bijv. "Voeg 'Python' toe aan je Skills sectie").

Real-time Monitoring: Ingebouwde sidebar met statistieken over sessie-duur, aantal analyses en geheugengebruik.

🛠️ Installatie
1. Vereisten
Zorg dat je Python 3.8 of hoger hebt geïnstalleerd.

2. Clone de repository
Bash
git clone <repository-url>
cd <project-map>

3. Installeer de afhankelijkheden
Bash
pip install streamlit spacy psutil
4. Download het SpaCy model
Dit project maakt gebruik van het medium-sized Engelse model voor semantische gelijkenis.

Bash
python -m spacy download en_core_web_md

Let op: De code verwacht dat het model beschikbaar is. Bij deployment op platforms zoals Streamlit Cloud kun je dit toevoegen aan je requirements.txt.

💻 Gebruik
Start de applicatie met het volgende commando:

Bash
streamlit run your_filename.py

Plak de tekst van je CV in het linker tekstveld.
Plak de Vacaturetekst in het rechter tekstveld.
Klik op "Analyze Resume vs. Job Description".
Bekijk je score en de suggesties om je CV te verbeteren.

🧠 Hoe het werkt
Preprocessing: Tekst wordt genormaliseerd (bijv. diploma's worden naar een standaardformaat omgezet).
Tokenizatie: SpaCy verwerkt de tekst naar noun_chunks en filtert op relevante POS-tags (Noun, Adj, Verb). Filtering: Stopwoorden en algemene termen (zoals "team player") worden gefilterd met een aangepaste lijst.
Similarity Check: Voor elk woord in de vacature wordt gecontroleerd of er een directe match is óf een semantische match (threshold $> 0.7$) in het CV.

📝 Logboek & Monitoring
De app houdt logs bij via de standaard Python logging module. Dit helpt bij het tracken van eventuele fouten tijdens het laden van het model of tijdens de extractie-fase.

Gemaakt als PoC voor intelligente CV-optimalisatie.
