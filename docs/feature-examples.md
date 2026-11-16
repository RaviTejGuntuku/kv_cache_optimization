# Feature examples and separating cases

Companion to JSON_features_to_support.docx. The [website companion](http://127.0.0.1:8771/json-features) highlights keywords. Source review is not a fresh decoder run. No proposed-parser results are claimed.

<a id="feature-1"></a>

## feature-1 — Declared required keys

Relevant keywords: `properties`, `required`, `additionalProperties`.


Schema:

```json
{
  "type": "object",
  "properties": {
    "id": {
      "type": "string"
    }
  },
  "required": [
    "id"
  ],
  "additionalProperties": false
}
```


Valid instance:

```json
{
  "id": "a"
}
```


Invalid instance:

```json
{}
```

<a id="feature-2"></a>

## feature-2 — Optional field omission

Relevant keywords: `required`.


Schema:

```json
{
  "type": "object",
  "properties": {
    "id": {
      "type": "string"
    },
    "note": {
      "type": "string"
    }
  },
  "required": [
    "id"
  ],
  "additionalProperties": false
}
```


Valid instance:

```json
{
  "id": "a"
}
```


Invalid instance:

```json
{
  "note": "n"
}
```

<a id="feature-3"></a>

## feature-3 — Required but undeclared key

Relevant keywords: `required`, `additionalProperties`.


Schema:

```json
{
  "type": "object",
  "properties": {},
  "required": [
    "id"
  ],
  "additionalProperties": {
    "type": "string"
  }
}
```


Valid instance:

```json
{
  "id": "a"
}
```


Invalid instance:

```json
{}
```

<a id="feature-4"></a>

## feature-4 — Typed extra keys

Relevant keywords: `additionalProperties`.


Schema:

```json
{
  "type": "object",
  "additionalProperties": {
    "type": "string"
  }
}
```


Valid instance:

```json
{
  "color": "red"
}
```


Invalid instance:

```json
{
  "color": 1
}
```

<a id="feature-5"></a>

## feature-5 — All key names constrained

Relevant keywords: `propertyNames`.


Schema:

```json
{
  "type": "object",
  "propertyNames": {
    "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"
  },
  "additionalProperties": {
    "type": "number"
  }
}
```


Valid instance:

```json
{
  "valid_key": 1
}
```


Invalid instance:

```json
{
  "bad-key": 1
}
```

<a id="feature-6"></a>

## feature-6 — Pattern keys plus extra keys

Relevant keywords: `patternProperties`, `additionalProperties`.


Schema:

```json
{
  "type": "object",
  "properties": {
    "label": {
      "type": "string"
    }
  },
  "patternProperties": {
    "^x_": {
      "type": "number"
    }
  },
  "additionalProperties": {
    "type": "string"
  }
}
```


Valid instance:

```json
{
  "x_count": 1,
  "note": "ok"
}
```


Invalid instance:

```json
{
  "x_count": "bad"
}
```

<a id="feature-7"></a>

## feature-7 — Overlapping key patterns

Relevant keywords: `patternProperties`.


Schema:

```json
{
  "type": "object",
  "patternProperties": {
    "^x": {
      "type": "string",
      "minLength": 2
    },
    "x$": {
      "type": "string",
      "pattern": "^[A-Z]+$"
    }
  },
  "additionalProperties": false
}
```


Valid instance:

```json
{
  "x": "AB"
}
```


Invalid instance:

```json
{
  "x": "ab"
}
```

<a id="feature-8"></a>

## feature-8 — Named key also matches pattern

Relevant keywords: `properties`, `patternProperties`.


Schema:

```json
{
  "type": "object",
  "properties": {
    "code": {
      "type": "string",
      "minLength": 2
    }
  },
  "patternProperties": {
    "^code$": {
      "pattern": "^[A-Z]+$"
    }
  },
  "additionalProperties": false
}
```


Valid instance:

```json
{
  "code": "AB"
}
```


Invalid instance:

```json
{
  "code": "ab"
}
```

<a id="feature-9"></a>

## feature-9 — Property count with optional fields

Relevant keywords: `minProperties`, `maxProperties`.


Schema:

```json
{
  "type": "object",
  "properties": {
    "a": {
      "type": "number"
    },
    "b": {
      "type": "number"
    },
    "c": {
      "type": "number"
    }
  },
  "additionalProperties": false,
  "minProperties": 2,
  "maxProperties": 2
}
```


Valid instance:

```json
{
  "a": 1,
  "b": 2
}
```


Invalid instance:

```json
{
  "a": 1
}
```

<a id="feature-10"></a>

## feature-10 — Presence dependencies

Relevant keywords: `dependentRequired`.


Schema:

```json
{
  "type": "object",
  "dependentRequired": {
    "profile": [
      "source"
    ]
  }
}
```


Valid instance:

```json
{
  "profile": "hd",
  "source": "movie.mp4"
}
```


Invalid instance:

```json
{
  "profile": "hd"
}
```

<a id="feature-11"></a>

## feature-11 — Close an extended object

Relevant keywords: `unevaluatedProperties`, `$ref`.


Schema:

```json
{
  "$defs": {
    "base": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string"
        }
      },
      "required": [
        "id"
      ]
    }
  },
  "$ref": "#/$defs/base",
  "properties": {
    "label": {
      "type": "string"
    }
  },
  "unevaluatedProperties": false
}
```


Valid instance:

```json
{
  "id": "a",
  "label": "A"
}
```


Invalid instance:

```json
{
  "id": "a",
  "extra": 1
}
```

<a id="feature-12"></a>

## feature-12 — Any key order with required identity

Relevant keywords: `required`.


Schema:

```json
{
  "type": "object",
  "properties": {
    "id": true,
    "name": true
  },
  "required": [
    "id",
    "name"
  ],
  "additionalProperties": false
}
```


Valid instance:

```json
{
  "name": "N",
  "id": "a"
}
```


Invalid instance:

```json
{
  "name": "N"
}
```

<a id="feature-13"></a>

## feature-13 — Unique dynamic key names

Relevant keywords: `additionalProperties`.

Duplicate-key serialization policy, not a schema assertion.


Schema:

```json
{
  "type": "object",
  "additionalProperties": {
    "type": "string"
  }
}
```


Valid instance:

```json
{
  "a": "1",
  "b": "2"
}
```


Invalid instance:

```json
{"a":"1","a":"2"}
```

<a id="feature-14"></a>

## feature-14 — Conditional required fields

Relevant keywords: `if`, `then`, `else`.


Schema:

```json
{
  "type": "object",
  "if": {
    "properties": {
      "status": {
        "const": "active"
      }
    },
    "required": [
      "status"
    ]
  },
  "then": {
    "required": [
      "publications"
    ],
    "properties": {
      "publications": {
        "type": "array",
        "minItems": 1
      }
    }
  }
}
```


Valid instance:

```json
{
  "status": "active",
  "publications": [
    "patent:x"
  ]
}
```


Invalid instance:

```json
{
  "status": "active"
}
```

<a id="feature-15"></a>

## feature-15 — Tagged alternative layouts

Relevant keywords: `oneOf`, `const`.


Schema:

```json
{
  "oneOf": [
    {
      "type": "object",
      "properties": {
        "kind": {
          "const": "circle"
        },
        "radius": {
          "type": "number"
        }
      },
      "required": [
        "kind",
        "radius"
      ]
    },
    {
      "type": "object",
      "properties": {
        "kind": {
          "const": "label"
        },
        "text": {
          "type": "string"
        }
      },
      "required": [
        "kind",
        "text"
      ]
    }
  ]
}
```


Valid instance:

```json
{
  "kind": "circle",
  "radius": 2
}
```


Invalid instance:

```json
{
  "kind": "circle",
  "text": "hi"
}
```

<a id="feature-16"></a>

## feature-16 — Combine object requirements

Relevant keywords: `allOf`.


Schema:

```json
{
  "type": "object",
  "allOf": [
    {
      "properties": {
        "name": {
          "type": "string"
        }
      },
      "required": [
        "name"
      ]
    },
    {
      "properties": {
        "age": {
          "type": "number",
          "minimum": 0
        }
      },
      "required": [
        "age"
      ]
    }
  ]
}
```


Valid instance:

```json
{
  "name": "Alice",
  "age": 30
}
```


Invalid instance:

```json
{
  "name": "Alice"
}
```

<a id="feature-17"></a>

## feature-17 — Mutually exclusive credentials

Relevant keywords: `not`, `required`.


Schema:

```json
{
  "type": "object",
  "properties": {
    "password": {
      "type": "string"
    },
    "token": {
      "type": "string"
    }
  },
  "not": {
    "required": [
      "password",
      "token"
    ]
  }
}
```


Valid instance:

```json
{
  "token": "abc"
}
```


Invalid instance:

```json
{
  "password": "pw",
  "token": "abc"
}
```

<a id="feature-18"></a>

## feature-18 — Pattern plus string length

Relevant keywords: `pattern`, `minLength`, `maxLength`.


Schema:

```json
{
  "type": "string",
  "pattern": "^[0-9]+$",
  "minLength": 9,
  "maxLength": 9
}
```


Valid instance:

```json
"123456789"
```


Invalid instance:

```json
"1234567890"
```

<a id="feature-19"></a>

## feature-19 — Finite matching markers

Relevant keywords: `pattern`.


Schema:

```json
{
  "type": "string",
  "pattern": "^%(s|i)%.+%\\1%$"
}
```


Valid instance:

```json
"%s%temperature%s%"
```


Invalid instance:

```json
"%s%temperature%i%"
```

<a id="feature-20"></a>

## feature-20 — Regular lookahead constraint

Relevant keywords: `pattern`.


Schema:

```json
{
  "type": "string",
  "pattern": "^(?=.*[A-Z])[A-Za-z]+$"
}
```


Valid instance:

```json
"Alice"
```


Invalid instance:

```json
"alice"
```

<a id="feature-21"></a>

## feature-21 — Enums constants and nullable values

Relevant keywords: `enum`, `type`.


Schema:

```json
{
  "type": [
    "string",
    "null"
  ],
  "enum": [
    "pending",
    "ready",
    null
  ]
}
```


Valid instance:

```json
null
```


Invalid instance:

```json
"other"
```

<a id="feature-23"></a>

## feature-23 — Array length including empty

Relevant keywords: `minItems`, `maxItems`.


Schema:

```json
{
  "type": "array",
  "items": {
    "type": "string"
  },
  "minItems": 0,
  "maxItems": 2
}
```


Valid instance:

```json
[]
```


Invalid instance:

```json
[
  "a",
  "b",
  "c"
]
```

<a id="feature-24"></a>

## feature-24 — Heterogeneous tuples

Relevant keywords: `prefixItems`, `items`.


Schema:

```json
{
  "type": "array",
  "prefixItems": [
    {
      "type": "string"
    },
    {
      "type": "number"
    }
  ],
  "items": false,
  "minItems": 2
}
```


Valid instance:

```json
[
  "score",
  1
]
```


Invalid instance:

```json
[
  "score",
  "one"
]
```

<a id="feature-25"></a>

## feature-25 — Unique array values

Relevant keywords: `uniqueItems`.


Schema:

```json
{
  "type": "array",
  "items": {
    "type": "string"
  },
  "uniqueItems": true
}
```


Valid instance:

```json
[
  "red",
  "blue"
]
```


Invalid instance:

```json
[
  "red",
  "red"
]
```

<a id="feature-26"></a>

## feature-26 — Matching array item counts

Relevant keywords: `contains`, `minContains`, `maxContains`.


Schema:

```json
{
  "type": "array",
  "items": {
    "type": "string"
  },
  "contains": {
    "const": "admin"
  },
  "minContains": 1,
  "maxContains": 2
}
```


Valid instance:

```json
[
  "reader",
  "admin"
]
```


Invalid instance:

```json
[
  "reader",
  "editor"
]
```

<a id="feature-27"></a>

## feature-27 — Close a reused tuple

Relevant keywords: `unevaluatedItems`, `$ref`.


Schema:

```json
{
  "$defs": {
    "pair": {
      "type": "array",
      "prefixItems": [
        {
          "type": "string"
        },
        {
          "type": "number"
        }
      ],
      "minItems": 2
    }
  },
  "$ref": "#/$defs/pair",
  "unevaluatedItems": false
}
```


Valid instance:

```json
[
  "score",
  1
]
```


Invalid instance:

```json
[
  "score",
  1,
  2
]
```

<a id="feature-28"></a>

## feature-28 — Local recursive references

Relevant keywords: `$ref`.


Schema:

```json
{
  "$defs": {
    "node": {
      "type": "object",
      "properties": {
        "value": {
          "type": "string"
        },
        "children": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/node"
          }
        }
      },
      "required": [
        "value"
      ]
    }
  },
  "$ref": "#/$defs/node"
}
```


Valid instance:

```json
{
  "value": "root",
  "children": [
    {
      "value": "leaf"
    }
  ]
}
```


Invalid instance:

```json
{
  "value": "root",
  "children": [
    1
  ]
}
```

<a id="feature-29"></a>

## feature-29 — Bundled URI references

Relevant keywords: `$id`, `$ref`.


Schema:

```json
{
  "$defs": {
    "Address": {
      "$id": "https://example.org/Address",
      "type": "object",
      "properties": {
        "city": {
          "type": "string"
        }
      },
      "required": [
        "city"
      ]
    }
  },
  "$ref": "https://example.org/Address"
}
```


Valid instance:

```json
{
  "city": "Boston"
}
```


Invalid instance:

```json
{
  "city": 1
}
```

<a id="feature-30"></a>

## feature-30 — Dynamic recursive references

Relevant keywords: `$dynamicRef`, `$dynamicAnchor`.


Schema:

```json
{
  "$dynamicAnchor": "node",
  "type": "object",
  "properties": {
    "children": {
      "type": "array",
      "items": {
        "$dynamicRef": "#node"
      }
    }
  },
  "additionalProperties": false
}
```


Valid instance:

```json
{
  "children": []
}
```


Invalid instance:

```json
{
  "children": [
    1
  ]
}
```

<a id="feature-31"></a>

## feature-31 — Integer type without arithmetic

Relevant keywords: `type`.


Schema:

```json
{
  "type": "integer"
}
```


Valid instance:

```json
7
```


Invalid instance:

```json
7.5
```

<a id="feature-32"></a>

## feature-32 — Boolean type

Relevant keywords: `type`.


Schema:

```json
{
  "type": "boolean"
}
```


Valid instance:

```json
true
```


Invalid instance:

```json
"true"
```

<a id="feature-33"></a>

## feature-33 — Fixed scalar constant

Relevant keywords: `const`.


Schema:

```json
{
  "type": "string",
  "const": "ok"
}
```


Valid instance:

```json
"ok"
```


Invalid instance:

```json
"error"
```

<a id="feature-34"></a>

## feature-34 — Simple anyOf union

Relevant keywords: `anyOf`.


Schema:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "boolean"
    }
  ]
}
```


Valid instance:

```json
"ok"
```


Invalid instance:

```json
1
```

<a id="feature-35"></a>

## feature-35 — Regular string pattern alone

Relevant keywords: `pattern`.


Schema:

```json
{
  "type": "string",
  "pattern": "^[A-Z]+$"
}
```


Valid instance:

```json
"AB"
```


Invalid instance:

```json
"ab"
```

<a id="feature-36"></a>

## feature-36 — String length alone

Relevant keywords: `minLength`, `maxLength`.


Schema:

```json
{
  "type": "string",
  "minLength": 2,
  "maxLength": 4
}
```


Valid instance:

```json
"ab"
```


Invalid instance:

```json
"a"
```

<a id="feature-37"></a>

## feature-37 — Homogeneous array item types

Relevant keywords: `items`.


Schema:

```json
{
  "type": "array",
  "items": {
    "type": "string"
  }
}
```


Valid instance:

```json
[
  "a",
  "b"
]
```


Invalid instance:

```json
["a",1]
```

<a id="feature-38"></a>

## feature-38 — Presence-only dependentSchemas

Relevant keywords: `dependentSchemas`.


Schema:

```json
{
  "type": "object",
  "dependentSchemas": {
    "profile": {
      "required": [
        "source"
      ]
    }
  }
}
```


Valid instance:

```json
{
  "profile": "hd",
  "source": "f"
}
```


Invalid instance:

```json
{"profile":"hd"}
```

<a id="feature-39"></a>

## feature-39 — Selected format assertions

Relevant keywords: `format`.

The invalid witness assumes opt-in format assertion.


Schema:

```json
{
  "type": "string",
  "format": "date-time"
}
```


Valid instance:

```json
"2026-11-15T12:00:00Z"
```


Invalid instance:

```json
"not-a-date"
```

<a id="feature-40"></a>

## feature-40 — Numeric assertions inventory

Relevant keywords: `minimum`, `maximum`, `multipleOf`.


Schema:

```json
{
  "type": "number",
  "minimum": 0,
  "maximum": 100,
  "multipleOf": 0.5
}
```


Valid instance:

```json
1.5
```


Invalid instance:

```json
1.2
```

<a id="S1"></a>

## S1 — Active scientific record

Relevant keywords: `if`, `then`, `else`.


Schema:

```json
{
  "type": "object",
  "properties": {
    "status": {
      "enum": [
        "active",
        "draft"
      ]
    },
    "publications": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 1,
      "maxItems": 2
    }
  },
  "required": [
    "status"
  ],
  "additionalProperties": false,
  "if": {
    "properties": {
      "status": {
        "const": "active"
      }
    },
    "required": [
      "status"
    ]
  },
  "then": {
    "required": [
      "publications"
    ]
  }
}
```


Valid instance:

```json
{
  "status": "active",
  "publications": [
    "p1"
  ]
}
```


Invalid instance:

```json
{"status":"active"}
```

<a id="S2"></a>

## S2 — Profile requires source

Relevant keywords: `dependentRequired`.


Schema:

```json
{
  "type": "object",
  "properties": {
    "profile": {
      "type": "string"
    },
    "source": {
      "type": "string"
    }
  },
  "additionalProperties": false,
  "dependentRequired": {
    "profile": [
      "source"
    ]
  }
}
```


Valid instance:

```json
{
  "profile": "hd",
  "source": "f"
}
```


Invalid instance:

```json
{"profile":"hd"}
```

<a id="S3"></a>

## S3 — One credential method

Relevant keywords: `not`, `required`.


Schema:

```json
{
  "type": "object",
  "properties": {
    "password": {
      "type": "string"
    },
    "token": {
      "type": "string"
    }
  },
  "minProperties": 1,
  "additionalProperties": false,
  "not": {
    "required": [
      "password",
      "token"
    ]
  }
}
```


Valid instance:

```json
{
  "token": "t"
}
```


Invalid instance:

```json
{"password":"p","token":"t"}
```

<a id="S4"></a>

## S4 — Distinct role values

Relevant keywords: `uniqueItems`.


Schema:

```json
{
  "type": "array",
  "items": {
    "enum": [
      "admin",
      "reader",
      "editor"
    ]
  },
  "minItems": 1,
  "maxItems": 3,
  "uniqueItems": true
}
```


Valid instance:

```json
[
  "admin",
  "reader"
]
```


Invalid instance:

```json
["reader","reader"]
```

<a id="S5"></a>

## S5 — At least one administrator

Relevant keywords: `contains`, `minContains`, `maxContains`.


Schema:

```json
{
  "type": "array",
  "items": {
    "enum": [
      "admin",
      "reader"
    ]
  },
  "minItems": 1,
  "maxItems": 3,
  "contains": {
    "const": "admin"
  },
  "minContains": 1
}
```


Valid instance:

```json
[
  "reader",
  "admin"
]
```


Invalid instance:

```json
["reader"]
```

<a id="S6"></a>

## S6 — At most one administrator

Relevant keywords: `contains`, `minContains`, `maxContains`.


Schema:

```json
{
  "type": "array",
  "items": {
    "enum": [
      "admin",
      "reader"
    ]
  },
  "minItems": 1,
  "maxItems": 3,
  "contains": {
    "const": "admin"
  },
  "minContains": 1,
  "maxContains": 1
}
```


Valid instance:

```json
[
  "admin",
  "reader"
]
```


Invalid instance:

```json
["admin","admin"]
```

<a id="S7"></a>

## S7 — Key matches two value rules

Relevant keywords: `patternProperties`.


Schema:

```json
{
  "type": "object",
  "patternProperties": {
    "^x": {
      "type": "string",
      "minLength": 2
    },
    "x$": {
      "type": "string",
      "pattern": "^[A-Z]+$"
    }
  },
  "additionalProperties": false
}
```


Valid instance:

```json
{
  "x": "AB"
}
```


Invalid instance:

```json
{"x":"ab"}
```

<a id="S8"></a>

## S8 — Dependency expressed as a subschema

Relevant keywords: `dependentSchemas`.


Schema:

```json
{
  "type": "object",
  "properties": {
    "profile": {
      "type": "string"
    },
    "source": {
      "type": "string"
    }
  },
  "additionalProperties": false,
  "dependentSchemas": {
    "profile": {
      "required": [
        "source"
      ]
    }
  }
}
```


Valid instance:

```json
{
  "profile": "hd",
  "source": "f"
}
```


Invalid instance:

```json
{"profile":"hd"}
```
