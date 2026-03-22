import codecs

try:
    with open('cookies.txt', 'rb') as f:
        raw = f.read()

    # Strip UTF-16 LE BOM (FF FE)
    if raw.startswith(b'\xff\xfe'):
        raw = raw[2:]
        text = raw.decode('utf-16le')
    # Strip UTF-8 BOM (EF BB BF)
    elif raw.startswith(codecs.BOM_UTF8):
        raw = raw[3:]
        text = raw.decode('utf-8')
    else:
        text = raw.decode('utf-8', errors='ignore')

    # Replaces CRLF with strict LF for Unix-style Netscape cookie validity
    text = text.replace('\r\n', '\n')

    with open('cookies.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    print("Fixed cookies.txt encoding successfully.")
except Exception as e:
    print(f"Error fixing cookies: {e}")
