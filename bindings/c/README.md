# Tokenizers C FFI Bindings

Pure C FFI bindings for the HuggingFace Tokenizers library, designed for .NET P/Invoke interop.

## Features

- No Python or Node.js runtime dependencies
- Pure C ABI exports
- Compatible with .NET P/Invoke
- Supports all major tokenizer algorithms (BPE, WordPiece, Unigram)

## Building

```bash
cargo build --release
```

This produces:

- **Linux**: `libtokenizers.so`
- **Windows**: `tokenizers.dll`
- **macOS**: `libtokenizers.dylib`

## C API

### Basic Usage

```c
// Load tokenizer from file
CTokenizer* tok = tokenizer_from_file("tokenizer.json");

// Encode text
size_t len;
uint32_t* ids = tokenizer_encode(tok, "Hello world", true, &len);

// Decode IDs
char* text = tokenizer_decode(tok, ids, len, true);

// Cleanup
tokenizer_free_string(text);
tokenizer_free_ids(ids, len);
tokenizer_free(tok);
```

## .NET P/Invoke Example

```csharp
using System;
using System.Runtime.InteropServices;

public class TokenizerWrapper
{
    [DllImport("tokenizers", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr tokenizer_from_file(string path);

    [DllImport("tokenizers", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr tokenizer_encode(
        IntPtr tokenizer,
        string text,
        bool addSpecialTokens,
        out UIntPtr outLen
    );

    [DllImport("tokenizers", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr tokenizer_decode(
        IntPtr tokenizer,
        uint[] ids,
        UIntPtr len,
        bool skipSpecialTokens
    );

    [DllImport("tokenizers", CallingConvention = CallingConvention.Cdecl)]
    private static extern void tokenizer_free(IntPtr tokenizer);

    [DllImport("tokenizers", CallingConvention = CallingConvention.Cdecl)]
    private static extern void tokenizer_free_ids(IntPtr ids, UIntPtr len);

    [DllImport("tokenizers", CallingConvention = CallingConvention.Cdecl)]
    private static extern void tokenizer_free_string(IntPtr str);
}
```

## License

Apache-2.0
