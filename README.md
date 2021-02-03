# ⚒ SciDart examples

Here you will find some examples of usage of SciDart.

## NumDart basics

### Arrays

```Dart
import 'package:scidart/numdart.dart';

void main() {
    // 1D array creation
    var a = Array([1.0, 2.0, 3.0]);

    print(a); // print vector

    // 2D array creation
    var mA = Array2d.empty();
    var line = Array([1.0, 2.0, 3.0]);

    mA.add(line);
    mA.add(line);
    mA.add(line);

    print(mA); // print matrix

    // 3D array creation
    var book = Array3d.empty();

    var page = Array2d.empty();
    page.add(Array([1.0, 2.0, 3.0]));
    page.add(Array([1.0, 2.0, 3.0]));
    page.add(Array([1.0, 2.0, 3.0]));

    book.add(page);
    book.add(page);
    book.add(page);

    print(book);

    // Create a array of complex numbers
    var aComplex = ArrayComplex([
      Complex(real: 2.0, imaginary: 2.0),
      Complex(real: 2.0, imaginary: 2.0),
      Complex(real: 2.0, imaginary: 2.0)
    ]);

    print(aComplex);
}
```

### Numeric spaces

```Dart
import 'package:scidart/numdart.dart';

void main() {
    var ar = arange(step: 0, stop: 100); // generate a array 0 to 99
    print(ar);
    
    var lin =
        linspace(0, 10, num: 40); // generate a array 0 to 10 with 40 elements
    print(lin);
    
    var on =
        ones(10); // generate a array with length 10 where all elements are 1
    print(on);
    
    var zer =
        zeros(5); // generate a array with length 5 where all elements are 0
    print(zer);
    
    // generate a 1Hz sine wave with sample frequency (fs) of 128Hz
    var N = 50.0;
    var fs = 128;
    var n = linspace(0, N, num: (N * fs).toInt(), endpoint: false);
    var f1 = 1; // 1Hz
    var sg = arraySin(arrayMultiplyToScalar(n, 2 * pi * f1));
    print(sg);
}
```

### Random data generation

```Dart
import 'package:scidart/numdart.dart';

void main() {
  // generate a array with 10 random elements 0 to 0.99
    var r = randomArray(10);
    print(r);
    
    // generate a 2D 3x3 with random elements
    var r2d = randomArray2d(3, 3);
    print(r2d);
    
    // generate ArrayComplex with 10 random elements
    var rAc = randomArrayComplex(10);
    print(rAc);
}
```

### Array operations

```Dart
import 'package:scidart/numdart.dart';

void main() {
    var a = Array([1.0, 2.0, 3.0]);
    var b = Array([1.0, 2.0, 3.0]);
    var aComplex = ArrayComplex([
      Complex(real: 2.0, imaginary: 2.0),
      Complex(real: 2.0, imaginary: 2.0),
      Complex(real: 2.0, imaginary: 2.0)
    ]);
    
    print(a + b);
    print(a - b);
    print(a * b);
    print(a / b);
    print(arrayConcat(a, b)); // array concatenation
    print(arrayMax(a));
    print(arrayMin(a));
    print(arrayDivisionToScalar(a, 2));
    print(a.getRangeArray(0, 2));
    print(arraySum(a));
    print(arrayToComplexArray(a));
    print(arrayPadStart(a, 2)); // add zeros at start of array
    print(arraySin(a)); // compute sin for all elements of array
    print(arrayReverse(a)); // array a reversed
    
    var c = Array([1.12121212, 2.12121212, 3.12121212]);
    print(arrayTruncateEachElement(c, 4,
        returnNewArray: true)); // truncate all values
    
    print(a == c);
    print(arrayPow(a, 2));
    print(arrayReshapeToMatrix(Array([1.0, 2.0, 3.0, 4.0]), 2));
    
    // Complex array operations
    print(arrayComplexAbs(aComplex));
    print(arrayComplexConjugate(aComplex));
    print(arrayComplexSum(aComplex, aComplex));
    print(arrayComplexPadStart(aComplex, 2));
    print(arrayComplexReverse(aComplex));
}
```

### Matrix operations

```Dart
import 'package:scidart/numdart.dart';

void main() {
    var mA = Array2d.empty();
    var line = Array([1.0, 2.0, 3.0]);
    
    mA.add(line);
    mA.add(line);
    mA.add(line);
    
    var mB = Array2d([
      Array([1.0, 2.0, 3.0]),
      Array([4.0, 5.0, 6.0]),
      Array([7.0, 8.0, 10.0])
    ]);
    var mC = Array2d([
      Array([1]),
      Array([2]),
      Array([3])
    ]);
    
    print(matrixDeterminant(mA)); // find matrix determinant
    print(matrixRank(mA)); // matrix rank
    print(matrixTranspose(mA)); // matrix transposition
    print(matrixInverse(mC)); // matrix inverse
    print(matrixDot(mA, mC)); // Matrix product
    print(matrixQR(mA).Q()); // Matrix Q of QR decomposition
    print(matrixSolve(mB, mC)); // solve a linear system
    
    var mD = Array2d([
      Array([1.121212121212]),
      Array([2.121212121212]),
      Array([3.121212121212])
    ]);
    array2dTruncateEachElement(mD, 4);
    print(mD);
    
    print(mA + mB);
    print(mA - mB);
    print(mA * mB);
    print(mA / mB);
    print(mA == mB);
    
    print(matrixNormOne(mA));
    print(matrixNormTwo(mA));
}
```

### Number helpers
```Dart
print(bitReverse(1, 3)); // rotate 000001 3 times and get 00100
print(highestOneBit(130)); // highest one bit. 0b10000010 => 1H: 2^8 = 128
print(intToBool(130)); // int to bool
print(boolToInt(false)); // bool to int
print(isEven(0)); // even
print(isOdd(1); // odd
print(isEvenDouble(1.1141, 3)); // check if double is even after truncation
print(isOddDouble(1.4474, 3)); // check if double is odd after truncation

var c1 = Complex(real: 1.1111, imaginary: 1);
var c2 = Complex(real: 1, imaginary: 1);

print(c1 + c2); // complex sum
print(c1 - c2); // complex sub
print(c1 * c2); // complex mul
print(c1 / c2); // complex div
print(complexAbs(c1)); // complex absolute
print(complexConjugate(c1)); // complex conjugate
print(complexCos(c1)); // complex cos
print(complexDivideScalar(c1, 2)); // complex div to scalar
print(complexTruncate(c1, 2)); // complex truncate
```

### Complex operations

```Dart
import 'package:scidart/numdart.dart';

void main() {
    var a = Complex(real: 1.121212, imaginary: 2.22222);
    var b = Complex.ri(4.0, 8.0);
    
    print(a + b);
    print(a - b);
    print(a * b);
    print(a / b);
    
    print(complexAbs(a));
    print(complexConjugate(a));
    print(complexCos(a));
    print(complexDivideScalar(a, 2));
    print(complexMultiplyScalar(a, 2));
    print(complexTruncate(a, 2));
}
```

### Calculus operations

```Dart
import 'package:scidart/numdart.dart';

void main() {
    var y = Array([-1, -1, -1, -1, 1, 1, 1, 1]); // function y points
    var x = arange(stop: 8); // function x points
    
    // array integration
    print(trapzArray(y, x: x)); // integration with trapezoidal rule
    print(simpsArray(y, x: x)); // integration with Simpson's rule
    
    // cumulative integration using trapezoidal rule, useful for signals
    print(cumIntegration(y, x: x));
    
    // function integration
    var f = (x) => pow(x, 2); // function definition: y = x^2
    var a = 0.0; // start integration interval
    var b = 10.0; // end integration interval
    var n = 10; // number of point between a and b
    print(trapzFunction(a, b, n, f)); // integration using trapezoidal rule
    print(simpsFunction(a, b, n, f)); // integration using Simpson's rule
    
    // array differentiation
    print(differentiateArray(y, x)); // return a array with the differentiation
    
    // function differentiation
    var px = 10.0; // point where the function will be differentiated
    // return the differentiation at point x=10.0
    print(differentiateFunction(px, f));
}
```

### Interpolation

```Dart
import 'package:scidart/numdart.dart';

void main() {
    // generate a x range -3 to 3
    var x = linspace(-3, 3, num: 30);
    // generate parabolic equation: y = 5 - 0.555*x^2
    var y = Array.fixed(x.length, initialValue: 5) -
        arrayMultiplyToScalar(arrayPow(x, 2), 0.555);
    // estimate the inter-sample points vx and vy around the maximum point
    // of y function
    var xMax = arrayArgMax(y);
    var yMax = arrayMax(y);
    
    print("Array points:");
    print("${xMax}, ${yMax}");
    
    print("Estimated points:");
    print(parabolic(y, xMax));
}
```

### Polynomial

```Dart
import 'package:scidart/numdart.dart';

void main() {
    // some point to estimate a polynomial regression
    var y = Array([0, 1, 4, 3, 2, 5, 0]); // y axis points
    var x = Array([0, 1, 2, 3, 4, 5, 6]); // x axis points
    var degree = 10; // suggested polynomial degree
    
    var p = PolyFit(x, y, degree); // polynomial regression
    print(p); // print polynomial
    
    // print the current polynomial degree, the function reduces the degree
    // to get the ideal degree rank for the actual points
    print(p.polyDegree());
    
    print(p.coefficients()); // polynomial coefficients
    
    print(p.predict(3.5)); // estimate a y value for the point x = 3.5
}
```

### Statistic

```Dart
import 'package:scidart/numdart.dart';

void main() {
    var a = randomArray(100); // generate a random array
    print(a); // print array
    print(mean(a)); // mean of array
    print(median(a)); // median of array
    print(mode(a)); // mode of array
    print(standardDeviation(a)); // standard deviation of array
    print(variance(a)); // variance of array
}
```

### Time

```Dart
import 'package:scidart/numdart.dart';

void main() {
    // the random function to measure the computation time
    var func = () {
      var x = arange(stop: 99);
      var y = randomArray(99);
      var py = PolyFit(x, y, 10);
      print(py);
      print(py.predict(55.5));
    };
    // the time measure the computation time if func and return
    var dur = timeit(func);
    print(dur); // print the computation time
}
```

## SciDart basics

### fftpack

```Dart
import 'package:scidart/numdart.dart';
import 'package:scidart/scidart.dart';

void main() {
    // generate the signal for test
    // 1Hz sine wave
    var N = 50.0;
    var fs = 128.0;
    var n = linspace(0, N, num: (N * fs).toInt(), endpoint: false);
    var f1 = 1.0; // 1Hz
    var sw = arraySin(arrayMultiplyToScalar(n, 2 * pi * f1));
    
    // return the fft of the signal
    var swF = fft(arrayToComplexArray(sw));
    print(swF);
    
    // return the real fft, in other words,
    // only the components with of positive frequencies
    var swFr = rfft(sw);
    print(swFr);
    
    // return the frequency scale of FFT in Hz
    var swFScale = fftFreq(swF.length, d: 1 / fs);
    print(swFScale);
    
    // return only the real frequencies of FFT, useful to use with rFFT
    swFScale = fftFreq(swF.length, d: 1 / fs, realFrequenciesOnly: true);
    print(swFScale);
    
    print(ifft(swF)); // inverse FFT
    print(rifft(swFr)); // inverse FFT of a real FFT
    
    print(dbfft(sw, fs)); // Return the frequency and fft in DB scale
}
```

### signal

```Dart
import 'package:scidart/numdart.dart';
import 'package:scidart/scidart.dart';

void main() {
    // sample signals for test
    var x = arrayConcat(zeros(8), ones(8));
    var y = arrayConcat(ones(8), zeros(8));
    
    //-------- convolution -----------//
    // simple numeric convolution
    print('convolution');
    print(convolution(x, y));
    
    // compute the convolution using FFT, that will be more fast only if:
    // x.length + y.length is equal power of 2;
    print(convolution(x, y, fast: true));
    
    // compute the convolution for complex signals using FFT
    print(convolutionComplex(arrayToComplexArray(x), arrayToComplexArray(y)));
    
    // compute the circular convolution for the complex signals
    print(convolutionCircularComplex(
        arrayToComplexArray(x), arrayToComplexArray(y)));

    // compute cross-relation for two singnals
    print(correlate(x, y))

    // compute cross-relation for two complex singnals
    correlateComplex(arrayToComplexArray(x), arrayToComplexArray(y));
    
    //-------- FIR filter -----------//
    // generate the test signal for filter
    // a sum of sine waves with 1Hz and 10Hz, 50 samples and
    // sample frequency (fs) 128Hz
    var N = 50.0;
    var fs = 128;
    var n = linspace(0, N, num: (N * fs).toInt(), endpoint: false);
    var f1 = 1; // 1Hz
    var sg1 = arraySin(arrayMultiplyToScalar(n, 2 * pi * f1));
    var f2 = 10; // 10Hz
    var sg2 = arraySin(arrayMultiplyToScalar(n, 2 * pi * f2));
    var sg = sg1 + sg2;
    
    // design a FIR filter low pass with cut frequency at 1Hz, the objective is
    // remove the 10Hz sine wave signal
    var nyq = 0.5 * fs; // nyquist frequency
    var fc = 1; // cut frequency 1Hz
    var normal_fc = fc / nyq; // frequency normalization for digital filters
    var numtaps = 30; // attenuation of the filter after cut frequency
    
    // generation of the filter coefficients
    var b = firwin(numtaps, Array([normal_fc]));
    print('FIR');
    print(b);
    
    //-------- Digital filter application -----------//
    print('digital filter application');
    // Apply the filter on the signal using lfilter function
    // lfilter uses direct form II transposed, for FIR filter
    // the a coefficient is 1.0
    var sgFiltered = lfilter(b, Array([1.0]), sg);
    
    print(sg); // show the original signal
    print(sgFiltered); // show the filtered signal
    
    //-------- peaks -----------//
    print('peaks');
    var pk = findPeaks(sgFiltered);
    print(pk[0]);// print the indexes of the peaks found in the signal
    print(pk[1]);// print the values of the peaks found in the signal
    
    //-------- window functions -----------//
    // generate a backmanharris window with x length
    print('window functions');
    var wBh = blackmanharris(x.length);
    print(x * wBh); // apply the windows on the signal
    
    // the windows available can be get with getWindow
    print(getWindow('hann', x.length)); // hann window
    
    // for kaiser window, we need pass beta (double value)
    print(getWindow(['kaiser', 2.0], x.length));
}
```

### special

```Dart
import 'package:scidart/numdart.dart';
import 'package:scidart/scidart.dart';

void main() {
    // modified Bessel function of order 0
    print(besselI0(10));
    
    // modified Bessel function of order 0 for a Array
    var a = randomArray(10);
    print(arrayBesselI0(a));
}
```

## Complete examples

### Avoid spectral leakage

```Dart
import 'package:scidart/numdart.dart';
import 'package:scidart/scidart.dart';

void main() {
    // generate the signals for test
    // 1Hz sine wave with incomplete with result a spectral leakage
    // in the frequency domain
    var N = 20.0;
    var fs = 128.0;
    var n = linspace(0, N, num: (N * fs).toInt(), endpoint: false);
    var f1 = 1.0; // 1Hz
    var sg = arraySin(arrayMultiplyToScalar(n, 2 * pi * f1));
    sg = sg.getRangeArray(0, sg.length - 201);
    sg = arrayConcat(sg, zeros(200));
    
    // reduce spectral leakage with a window function
    var sgWindowed = blackmanharris(sg.length) * sg;
    
    print(dbfft(sg, fs)); // fft of original signal
    print(dbfft(sgWindowed, fs)); // fft of windowed signal
}
```

### Frequency estimator

```Dart
import 'package:scidart/numdart.dart';
import 'package:scidart/scidart.dart';

void main() {
// generate the signals for test
    // 1Hz sine wave
    var N = 50.0;
    var fs = 128.0;
    var n = linspace(0, N, num: (N * fs).toInt(), endpoint: false);
    var f1 = 1.0; // 1Hz
    var sg1 = arraySin(arrayMultiplyToScalar(n, 2 * pi * f1));
    
    var fEstimated = freqFromFft(sg1, fs);
    
    print('The original and estimated frequency need be very close each other');
    print('Original frequency: ${f1}');
    print('Estimated frequency: ${fEstimated}');
    
    // both are equal if truncated
    expect(truncate(f1, 4), truncate(fEstimated, 4));
}

double freqFromFft(Array sig, double fs) {
  // Estimate frequency from peak of FFT

  // Compute Fourier transform of windowed signal
  // Avoid spectral leakage: https://en.wikipedia.org/wiki/Spectral_leakage
  var windowed = sig * blackmanharris(sig.length);
  var f = rfft(windowed);

  var fAbs = arrayComplexAbs(f);

  // Find the peak and interpolate to get a more accurate peak
  var i = arrayArgMax(fAbs); // Just use this for less-accurate, naive version

  // Parabolic approximation is necessary to get the exactly frequency of a discrete signal
  // since the frequency can be in some point between the samples.
  var true_i = parabolic(arrayLog(fAbs), i)[0];

  // Convert to equivalent frequency
  return fs * true_i / windowed.length;
}
```

## Examples **Scidart IO**

### CSV

```Dart
import 'package:scidart/numdart.dart';
import 'package:scidart_io/scidart_io.dart';

void main() async {
  // read stock_data.csv file in the same directory of the current
  // script
  // the delimiter of this file is ',' but can be any character
  // the reader skip one line in the header of the file and
  // one line at the end of the file
  var data = await readCSV('stock_data.csv', delimiter: ',', skipHeader: 1, skipFooter: 1);
  
  print(data); // show the data
  
  // generate a 2d array data for test
  var data2 = Array2d([
    Array([1, 2, 3, 4, 5]),
    Array([2, 3, 4, 5, 6]),
    Array([3, 4, 5, 6, 7]),
    Array([4, 5, 6, 7, 8]),
    Array([5, 6, 7, 8, 9]),
  ]);
  
  // define a file name
  var fileName = 'data_array.csv';
  
  // save the data in a CSV file
  await writeLinesCSV(data2, fileName);
  
  // read the same data again and convert to Array2d again
  var data2Read = await readCSV(fileName, convertToArray2d: true);
  
  // show the data
  print(data2Read);
}
```

### TXT

```Dart
import 'package:scidart/numdart.dart';
import 'package:scidart_io/scidart_io.dart';

void main() async {
  // define a data for tests
  var data = Array2d([
    Array([1, 2, 3, 4, 5]),
    Array([2, 3, 4, 5, 6]),
    Array([3, 4, 5, 6, 7]),
    Array([4, 5, 6, 7, 8]),
    Array([5, 6, 7, 8, 9]),
  ]);
  
  // define a file name
  var fileName = 'data_array.txt';
  
  // write lines in the txt files
  await writeLinesTxt(data.toString().split('\n'), fileName);
  
  // read the line again
  var dataRead = await readLinesTxt(fileName);
  
  // show in the terminal
  print(dataRead);
  
  // to write plain string in a file, just use writeTxt
  
  // write string in the txt files
  await writeTxt(data.toString(), fileName);
    
  // read a string again
  dataRead = await readTxt(fileName);
  
  print(dateRead);
}

```