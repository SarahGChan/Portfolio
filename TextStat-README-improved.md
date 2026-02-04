# textstat

Calculate readability statistics and complexity metrics for any text.

---

> **📝 Portfolio Documentation Sample**
> 
> This is a rewritten version of the textstat README created by Sarah Chan to demonstrate documentation improvement skills for potential employers. It is not part of the official textstat project. For the official textstat documentation, please visit the [textstat GitHub repository](https://github.com/textstat/textstat).

---

## Overview

textstat is a Python library that analyzes text readability using 20+ established formulas and metrics. Whether you're writing educational content, creating accessible documentation, or analyzing text complexity, textstat helps you measure how easy your text is to read and understand.

**Key Features:**
- **20+ readability formulas** including Flesch Reading Ease, Gunning Fog, SMOG Index
- **Multi-language support** with specialized formulas for Spanish, German, French, and more
- **Simple API** - one function call per metric
- **Zero dependencies** for basic functionality
- **Battle-tested** - used in content management systems, educational tools, and accessibility checkers

**Use Cases:**
- **Content Writers**: Ensure your articles match your target audience's reading level
- **Educators**: Assess if materials are appropriate for student grade levels
- **Accessibility Teams**: Verify compliance with plain language requirements
- **Technical Writers**: Measure documentation complexity and improve clarity

---

## Quick Start

### Installation

Install textstat via pip:

```bash
pip install textstat
```

### Basic Usage

```python
import textstat

text = """
The quick brown fox jumps over the lazy dog. 
This simple sentence demonstrates basic English usage.
"""

# Get readability score (0-100, higher = easier to read)
score = textstat.flesch_reading_ease(text)
print(f"Reading Ease: {score}")  # Output: ~80 (fairly easy)

# Get recommended grade level
grade = textstat.flesch_kincaid_grade(text)
print(f"Grade Level: {grade}")  # Output: ~3.5 (3rd-4th grade)
```

**Expected output:**
```
Reading Ease: 80.3
Grade Level: 3.5
```

---

## Choosing the Right Formula

Different formulas work better for different purposes. Here's a quick guide:

### For General Content (Recommended)

**Flesch Reading Ease** - Best for general audience content
```python
score = textstat.flesch_reading_ease(text)
# Returns: 0-100 (higher = easier)
# 90-100: Very Easy (5th grade)
# 60-70: Standard (8th-9th grade)
# 0-30: Very Difficult (college graduate)
```

**Flesch-Kincaid Grade** - Best for educational materials
```python
grade = textstat.flesch_kincaid_grade(text)
# Returns: U.S. grade level (e.g., 8.2 = 8th grade)
```

### For Academic & Technical Content

**Gunning Fog Index** - Identifies complex language
```python
fog = textstat.gunning_fog(text)
# Returns: Years of education needed
# Good for identifying "foggy" writing with too much jargon
```

**SMOG Index** - Accurate for complex texts
```python
smog = textstat.smog_index(text)
# Returns: Grade level (requires 30+ sentences)
# More accurate than Flesch for difficult texts
```

### For Quick Assessment

**Text Standard** - Consensus across multiple formulas
```python
standard = textstat.text_standard(text, float_output=False)
# Returns: "7th and 8th grade" (combines multiple metrics)
```

### Formula Comparison

| Formula | Best For | Range | Language Support |
|---------|----------|-------|------------------|
| Flesch Reading Ease | General content, blog posts | 0-100 | English, Spanish |
| Flesch-Kincaid Grade | Educational materials | Grade 1-16+ | English, Spanish |
| Gunning Fog | Business writing, documentation | Grade 6-17+ | English |
| SMOG Index | Academic papers, research | Grade 4-18 | English |
| Coleman-Liau | Materials with technical terms | Grade 1-16+ | English |
| Dale-Chall | Children's books, learning materials | Grade 1-16+ | English |

---

## Common Formulas

### Readability Scores

#### flesch_reading_ease()

Measures text readability on a 0-100 scale.

```python
score = textstat.flesch_reading_ease(text)
```

**Returns:** `float` - Readability score
- 90-100: Very easy (5th grade)
- 80-89: Easy (6th grade)
- 70-79: Fairly easy (7th grade)
- 60-69: Standard (8th-9th grade)
- 50-59: Fairly difficult (10th-12th grade)
- 30-49: Difficult (college)
- 0-29: Very confusing (college graduate)

**Example:**
```python
easy_text = "The cat sat on the mat. It was warm."
hard_text = "The feline positioned itself upon the textile floor covering, experiencing thermal comfort."

print(textstat.flesch_reading_ease(easy_text))   # ~95 (very easy)
print(textstat.flesch_reading_ease(hard_text))   # ~10 (very difficult)
```

---

#### flesch_kincaid_grade()

Converts readability to U.S. grade level.

```python
grade = textstat.flesch_kincaid_grade(text)
```

**Returns:** `float` - Grade level (e.g., 7.4 = 7th grade, 4 months)

**Example:**
```python
text = "Python is a programming language. It is easy to learn."
grade = textstat.flesch_kincaid_grade(text)
print(f"Grade Level: {grade}")  # ~4.1 (4th grade)
```

---

#### gunning_fog()

Estimates years of formal education needed to understand text.

```python
fog = textstat.gunning_fog(text)
```

**Returns:** `float` - Grade level based on complex words

**Formula focuses on:**
- Sentence length
- Percentage of words with 3+ syllables

**Best for:** Business writing, technical documentation

**Example:**
```python
simple = "We need to talk. The project is behind schedule."
complex = "It is imperative that we convene immediately to discuss the considerable delays affecting deliverables."

print(textstat.gunning_fog(simple))   # ~4.8
print(textstat.gunning_fog(complex))  # ~14.2
```

---

#### smog_index()

Estimates grade level needed to understand text (requires 30+ sentences).

```python
smog = textstat.smog_index(text)
```

**Returns:** `float` - Grade level

**Note:** Requires at least 30 sentences for statistical validity. Returns error for shorter texts.

**Best for:** Academic papers, research documents

---

### Text Statistics

#### Basic Counts

```python
# Character and word counts
chars = textstat.char_count(text)                    # Total characters
chars_no_spaces = textstat.char_count(text, ignore_spaces=True)
letters = textstat.letter_count(text)                # Letters only (no punctuation)
words = textstat.lexicon_count(text)                 # Total words
sentences = textstat.sentence_count(text)            # Total sentences
```

#### Syllable Counts

```python
syllables = textstat.syllable_count(text)            # Total syllables
polysyllables = textstat.polysyllabcount(text)       # Words with 3+ syllables
monosyllables = textstat.monosyllabcount(text)       # Single-syllable words
```

#### Complexity Metrics

```python
difficult = textstat.difficult_words(text)           # Words not on easy word list
long_words = textstat.long_word_count(text)          # Words with 7+ characters
```

---

## Advanced Usage

### Setting Language

By default, textstat uses English (en_US). Change language for accurate syllable counting:

```python
textstat.set_lang('es')  # Spanish
score = textstat.flesch_reading_ease(spanish_text)
```

**Supported languages:**
- `en` or `en_US` - English (default)
- `es` - Spanish
- `de` - German
- `fr` - French
- `it` - Italian
- `nl` - Dutch
- `pl` - Polish
- `ru` - Russian

### Spanish-Specific Formulas

Special readability formulas optimized for Spanish text:

```python
textstat.set_lang('es')

# Fernández Huerta (Spanish adaptation of Flesch)
score = textstat.fernandez_huerta(spanish_text)

# Szigriszt-Pazos
score = textstat.szigriszt_pazos(spanish_text)

# Gutiérrez de Polini
score = textstat.gutierrez_polini(spanish_text)

# Crawford
score = textstat.crawford(spanish_text)
```

### Batch Processing

Analyze multiple documents efficiently:

```python
documents = {
    "article_1.txt": "First article content...",
    "article_2.txt": "Second article content...",
    "article_3.txt": "Third article content..."
}

results = {}
for name, content in documents.items():
    results[name] = {
        'flesch': textstat.flesch_reading_ease(content),
        'grade': textstat.flesch_kincaid_grade(content),
        'difficult_words': textstat.difficult_words(content)
    }

# Find most readable document
most_readable = max(results.items(), key=lambda x: x[1]['flesch'])
print(f"Most readable: {most_readable[0]}")
```

---

## Real-World Examples

### Example 1: Content Management System

Ensure blog posts match target audience reading level:

```python
def check_readability(article_text, target_grade=8):
    """
    Validate article readability for general audience.
    
    Args:
        article_text: Article content
        target_grade: Maximum acceptable grade level
        
    Returns:
        dict: Readability metrics and pass/fail status
    """
    grade = textstat.flesch_kincaid_grade(article_text)
    ease = textstat.flesch_reading_ease(article_text)
    
    return {
        'grade_level': grade,
        'reading_ease': ease,
        'passes': grade <= target_grade,
        'recommendation': 'Simplify' if grade > target_grade else 'Good to publish'
    }

# Usage
article = "Your article content here..."
result = check_readability(article, target_grade=8)

if not result['passes']:
    print(f"Warning: Grade level {result['grade_level']} exceeds target")
    print(f"Recommendation: {result['recommendation']}")
```

### Example 2: Documentation Validation

Check technical documentation complexity:

```python
def analyze_documentation(doc_text):
    """Comprehensive documentation analysis."""
    
    return {
        'grade_level': textstat.flesch_kincaid_grade(doc_text),
        'reading_ease': textstat.flesch_reading_ease(doc_text),
        'fog_index': textstat.gunning_fog(doc_text),
        'difficult_words': textstat.difficult_words(doc_text),
        'total_words': textstat.lexicon_count(doc_text),
        'sentences': textstat.sentence_count(doc_text),
        'avg_sentence_length': textstat.lexicon_count(doc_text) / textstat.sentence_count(doc_text)
    }

# Generate report
stats = analyze_documentation(documentation)
print(f"Grade Level: {stats['grade_level']}")
print(f"Difficult Words: {stats['difficult_words']} ({stats['difficult_words']/stats['total_words']*100:.1f}%)")
print(f"Average Sentence Length: {stats['avg_sentence_length']:.1f} words")
```

### Example 3: A/B Testing Headlines

Compare headline readability for better engagement:

```python
headlines = [
    "5 Simple Ways to Improve Your Writing",
    "Quintessential Methodologies for Augmenting Compositional Proficiency"
]

for i, headline in enumerate(headlines, 1):
    score = textstat.flesch_reading_ease(headline)
    grade = textstat.flesch_kincaid_grade(headline)
    print(f"Headline {i}:")
    print(f"  Readability: {score:.1f}")
    print(f"  Grade Level: {grade:.1f}")
    print()

# Output shows Headline 1 is much more accessible
```

---

## Troubleshooting

### "Not enough sentences" Error (SMOG Index)

**Problem:** `smog_index()` requires at least 30 sentences for statistical validity.

**Solution:** Use alternative formulas for shorter texts:
```python
try:
    score = textstat.smog_index(text)
except:
    # Use Flesch-Kincaid instead for short texts
    score = textstat.flesch_kincaid_grade(text)
```

### Unexpectedly High Difficulty Score

**Problem:** Formula returns much higher difficulty than expected.

**Common causes:**
1. **Long sentences** - Break into shorter sentences
2. **Complex words** - Use simpler alternatives
3. **Technical jargon** - Consider your audience
4. **Passive voice** - Switch to active voice

**Debug approach:**
```python
# Identify the problem
print(f"Avg sentence length: {textstat.lexicon_count(text) / textstat.sentence_count(text):.1f} words")
print(f"Difficult words: {textstat.difficult_words(text)}")
print(f"Polysyllables: {textstat.polysyllabcount(text)}")

# Sentences over 20 words often increase difficulty
# More than 10% difficult words raises scores
```

### Inconsistent Scores Across Formulas

**Problem:** Different formulas give different grade levels.

**Explanation:** This is normal. Each formula emphasizes different factors:
- **Flesch-Kincaid**: Sentence length + syllables
- **Gunning Fog**: Complex words (3+ syllables)
- **SMOG**: Polysyllable density
- **Coleman-Liau**: Characters per word

**Solution:** Use `text_standard()` for consensus:
```python
# Get agreement across multiple formulas
standard = textstat.text_standard(text)
print(standard)  # "7th and 8th grade"
```

### Non-English Text Returns Incorrect Scores

**Problem:** Readability scores seem wrong for non-English text.

**Solution:** Set the language before analysis:
```python
# Wrong - uses English syllable counting on Spanish text
score = textstat.flesch_reading_ease(spanish_text)  # Inaccurate

# Correct - uses Spanish language rules
textstat.set_lang('es')
score = textstat.flesch_reading_ease(spanish_text)  # Accurate
```

---

## How Readability Formulas Work

Understanding what these formulas measure helps you improve your writing.

### Common Factors

Most readability formulas consider:

1. **Sentence Length**
   - Longer sentences = harder to read
   - Target: 15-20 words per sentence for general audience

2. **Word Length / Syllables**
   - More syllables = harder words
   - Complex words reduce readability

3. **Word Familiarity**
   - Technical terms and rare words increase difficulty
   - Dale-Chall uses a list of 3,000 common words

### Formula Breakdown

**Flesch Reading Ease:**
```
Score = 206.835 - (1.015 × ASL) - (84.6 × ASW)

Where:
  ASL = Average Sentence Length (words/sentence)
  ASW = Average Syllables per Word
```

**Flesch-Kincaid Grade Level:**
```
Grade = (0.39 × ASL) + (11.8 × ASW) - 15.59
```

**Gunning Fog Index:**
```
Grade = 0.4 × [(ASL) + 100 × (Complex Words / Total Words)]

Where:
  Complex Words = words with 3+ syllables
```

---

## Limitations & Considerations

### What Readability Formulas Don't Measure

❌ **Content quality** - A readable text can still be poorly written  
❌ **Accuracy** - Simple words can convey complex ideas incorrectly  
❌ **Audience knowledge** - Technical audiences may prefer technical terms  
❌ **Context** - Some complex writing is appropriate (legal, medical)  

### Best Practices

✅ Use formulas as **guidance**, not absolute rules  
✅ Consider your **target audience**  
✅ Combine metrics with **human review**  
✅ Test with **actual readers** when possible  
✅ Remember: **Clear ≠ Simplistic**

### When to Ignore Scores

- **Specialized audiences**: Technical docs for experts
- **Creative writing**: Poetry, literature with intentional complexity
- **Legal/medical**: Precision sometimes requires complexity
- **Non-prose**: Code examples, lists, structured data

---

## API Reference

For complete parameter documentation and return values, see the [Full API Reference](api-reference.md).

**Quick Reference:**

| Method | Returns | Purpose |
|--------|---------|---------|
| `flesch_reading_ease(text)` | float (0-100) | General readability |
| `flesch_kincaid_grade(text)` | float | U.S. grade level |
| `gunning_fog(text)` | float | Grade level (complex words) |
| `smog_index(text)` | float | Grade level (30+ sentences) |
| `coleman_liau_index(text)` | float | Grade level (character-based) |
| `automated_readability_index(text)` | float | Grade level |
| `dale_chall_readability_score(text)` | float | Grade level (word familiarity) |
| `difficult_words(text)` | int | Count of difficult words |
| `linsear_write_formula(text)` | float | Grade level |
| `text_standard(text)` | str | Consensus grade level |

---

## Contributing

Contributions are welcome! Please see the [Contributing Guidelines](CONTRIBUTING.md) for details.

**Areas where help is needed:**
- Additional language support
- New readability formulas
- Performance improvements
- Documentation improvements

---

## Resources

### About Readability Formulas

- **Flesch Reading Ease**: [Research Paper](https://example.com/flesch)
- **Gunning Fog Index**: [Original Publication](https://example.com/fog)
- **Plain Language Guidelines**: [PlainLanguage.gov](https://example.com/plain-language)

### Related Tools

- **Hemingway Editor**: Desktop app for readability improvement
- **Readable.com**: Online readability checker
- **Grammarly**: Writing assistant with readability metrics

---

## License

textstat is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Support

- **Issues**: [Report bugs or request features](https://github.com/project/repo/issues)
- **Documentation**: [Full documentation](https://docs.example.com)
- **Community**: [Discussions forum](https://community.example.com)

---

**Last Updated:** January 2026  
**Version:** 0.7.0
