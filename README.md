# Odpowiedzi na część teoretyczną

## Opis architektury sieci

Rozważana sieć neuronowa składa się z:
- dwóch cech wejściowych: $x_1, x_2$,
- jednej warstwy ukrytej złożonej z 2 neuronów z funkcją aktywacji ReLU,
- jednego neuronu wyjściowego bez funkcji aktywacji.

Dla pojedynczej obserwacji dane wejściowe wynoszą:
- $x_1 = 2$
- $x_2 = 3$
- wartość docelowa: $y = 5$

Funkcją straty jest Mean Squared Error (MSE):  
$L = (\hat{y} - y)^2$

---

## Model matematyczny sieci

### Warstwa ukryta

Dla pierwszego neuronu:  
$z_1 = w_{11}x_1 + w_{12}x_2 + b_1$

Dla drugiego neuronu:  
$z_2 = w_{21}x_1 + w_{22}x_2 + b_2$

Funkcja aktywacji ReLU:  
$a_i = \max(0, z_i)$

### Warstwa wyjściowa

$\hat{y} = v_1 a_1 + v_2 a_2 + b_3$

---

## Przypadek 1: inicjalizacja wszystkich parametrów wartością 0.0

### Forward pass

$$
z_1 = z_2 = 0
$$

$$
a_1 = a_2 = \text{ReLU}(0) = 0
$$

$$
\hat{y} = 0
$$

$$
L = (0 - 5)^2 = 25
$$

## Pochodne

Pochodna funkcji straty względem wyjścia:  
$\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y) = -10$

Pochodna funkcji ReLU w punkcie $z = 0$:  
$\frac{d}{dz}\text{ReLU}(0) = 0$

W rezultacie wszystkie gradienty wag i biasów są równe zero:  
$\frac{\partial L}{\partial w} = 0,\quad \frac{\partial L}{\partial b} = 0$

## Wniosek

Sieć neuronowa nie jest w stanie się uczyć, ponieważ brak jest niezerowych gradientów umożliwiających aktualizację parametrów.

---

## Przypadek 2: inicjalizacja wszystkich parametrów wartością 1.0

### Forward pass

Warstwa ukryta:  
$z_1 = 1 \cdot 2 + 1 \cdot 3 + 1 = 6$  

$z_2 = 6$

Po zastosowaniu ReLU:  
$a_1 = a_2 = 6$

Wyjście sieci:  
$\hat{y} = 1 \cdot 6 + 1 \cdot 6 + 1 = 13$

Strata:  
$L = (13 - 5)^2 = 64$


---

## Backpropagation – pochodne

Pochodna funkcji straty względem wyjścia:  
$\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y) = 16$

Gradienty wag warstwy wyjściowej:  
$\frac{\partial L}{\partial v_1} = 16 \cdot 6 = 96$  

$\frac{\partial L}{\partial v_2} = 96$

Gradient biasu neuronu wyjściowego:  
$\frac{\partial L}{\partial b_3} = 16$

Gradienty wag warstwy ukrytej:  
$\frac{\partial L}{\partial w_{11}} = 16 \cdot 1 \cdot 2 = 32$  

$\frac{\partial L}{\partial w_{12}} = 16 \cdot 1 \cdot 3 = 48$

Analogicznie wyznacza się pochodne dla drugiego neuronu warstwy ukrytej.

### Wniosek

Sieć neuronowa jest w stanie się uczyć, ponieważ gradienty są różne od zera.

---

## Zastosowanie sieci neuronowych

Sieci neuronowe warto stosować w zadaniach, w których występują:
- złożone i nieliniowe zależności pomiędzy danymi,
- dane zaszumione,
- bardzo duża liczba możliwych kombinacji danych wejściowych.

Nie jest możliwe stworzenie programu opartego wyłącznie na instrukcjach warunkowych typu if, ponieważ liczba potencjalnych przypadków byłaby zbyt duża, a zależności pomiędzy zmiennymi nie są znane z góry. Sieci neuronowe uczą się wzorców bez konieczności ręcznego definiowania reguł.

---

## Rola funkcji aktywacji

Funkcje aktywacji wprowadzają nieliniowość do modelu. W przypadku usunięcia funkcji aktywacji z sieci wielowarstwowej, cała sieć sprowadza się do jednej funkcji liniowej, niezależnie od liczby warstw, co znacząco ogranicza jej zdolności modelowania złożonych relacji.

---

## Rola dropout’u

Dropout jest techniką regularizacji polegającą na losowym wyłączaniu neuronów podczas treningu. Zapobiega to przeuczeniu modelu oraz poprawia zdolność generalizacji sieci neuronowej na nowe dane.
