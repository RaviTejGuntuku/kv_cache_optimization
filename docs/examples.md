# Illustrative JSON Schema examples

Numbering matches the [coverage table](coverage.md). Row 22 (numeric constraints) is omitted. Row 13 illustrates the study’s duplicate-key policy rather than a schema assertion. Source-derived examples are not additional decoder measurements.

<a id="row-1"></a>

## 1. Declared required keys

An object must have a string id and no extra fields.

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

<a id="row-2"></a>

## 2. Optional field omission

Keep id, but allow note to be omitted.

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

<a id="row-3"></a>

## 3. Required but undeclared key

Require id even though only the extra-key schema defines its type.

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

<a id="row-4"></a>

## 4. Typed extra keys

Allow arbitrary keys, but require string values.

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

<a id="row-5"></a>

## 5. All key names constrained

Every key must be a valid identifier: valid_key passes; bad-key fails.

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

<a id="row-6"></a>

## 6. Pattern keys plus extra keys

x_count must be a number; other extra fields must be strings.

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

<a id="row-7"></a>

## 7. Overlapping key patterns

x matches two patterns: its string must be at least two characters AND uppercase.

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

<a id="row-8"></a>

## 8. Named key also matches pattern

code must satisfy its named-field rule AND its matching uppercase pattern.

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

<a id="row-9"></a>

## 9. Property count with optional fields

Choose exactly two of a, b and c.

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

<a id="row-10"></a>

## 10. Presence dependencies

If profile is present, source must also be present.

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

<a id="row-11"></a>

## 11. Close an extended object

Extend a reusable id object with label, then reject all other keys.

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

<a id="row-12"></a>

## 12. Any key order with required identity

Accept name then id as well as id then name, without losing required keys.

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

<a id="row-13"></a>

## 13. Unique dynamic key names

Allow a dynamic map, but reject repeated object key names.

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

Invalid instance under the duplicate-key policy:

```json
{"a":"1","a":"2"}
```

<a id="row-14"></a>

## 14. Conditional required fields

An active record must include a nonempty publications array.

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

<a id="row-15"></a>

## 15. Tagged alternative layouts

A required tag selects a circle layout or a text-label layout.

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

<a id="row-16"></a>

## 16. Combine object requirements

Combine reusable object requirements with allOf.

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

<a id="row-17"></a>

## 17. Mutually exclusive credentials

Allow password or token, but never both together.

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

<a id="row-18"></a>

## 18. Pattern plus string length

An organization code must contain digits and have exactly nine characters.

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

<a id="row-19"></a>

## 19. Finite matching markers

The closing marker must equal the opening marker, chosen from s or i.

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

<a id="row-20"></a>

## 20. Regular lookahead constraint

Allow letters only and require at least one uppercase letter.

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

<a id="row-21"></a>

## 21. Enums constants and nullable values

Allow only pending, ready or null.

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

<a id="row-23"></a>

## 23. Array length including empty

Allow zero to two strings, including an empty array.

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

<a id="row-24"></a>

## 24. Heterogeneous tuples

Require a two-position tuple with different value types.

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

<a id="row-25"></a>

## 25. Unique array values

Reject repeated array values, such as ["red","red"].

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

<a id="row-26"></a>

## 26. Matching array item counts

Require one or two occurrences of admin in a string array.

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

<a id="row-27"></a>

## 27. Close a reused tuple

Reuse a tuple schema and forbid unevaluated trailing items.

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

<a id="row-28"></a>

## 28. Local recursive references

A tree node may contain children of the same schema.

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

<a id="row-29"></a>

## 29. Bundled URI references

Resolve a bundled Address schema by its URI identifier.

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

<a id="row-30"></a>

## 30. Dynamic recursive references

Resolve recursive children through a dynamic anchor.

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
