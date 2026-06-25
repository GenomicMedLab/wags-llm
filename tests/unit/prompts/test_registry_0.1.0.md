---
name: test-registry
description: A helpful assistant that processes text input and returns a JSON
  object with a value field set to 1.
---

# Test Skill

## Overview

A simple skill that processes text input and returns a structured JSON response.

## When to Use

Use when you need to process a text input and receive a standardized JSON output with a value field.

## Instructions

You are a helpful assistant. Return a JSON object with a `value` field set to `1`.

## Input Format

The input will be a JSON object with the following structure:

```json
{
    "text": "the text to process"
}
```

## Output Format

Return a JSON object matching the provided schema:

```json
{
    "value": 1
}
```

## Examples

### Input
```json
{
    "text": "hello"
}
```

### Output
```json
{
    "value": 1
}
```
