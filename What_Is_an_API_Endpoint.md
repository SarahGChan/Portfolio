# What Is an API Endpoint?

## Audience

This guide is for frontend or full-stack developers who are new to working directly with REST APIs. It explains what an API endpoint is, how it is structured, and how to interact with it using HTTP requests.

------------------------------------------------------------------------

## Overview

When you work with a REST API, you interact with specific URLs that allow your application to retrieve or modify data. These URLs are called **endpoints**. An API endpoint represents a specific resource or action that a server makes available to clients.

Each endpoint is defined by a combination of:

-   A URL path (such as `/users/123`)
-   An HTTP method (such as `GET` or `POST`)
-   Expected input (path parameters, query parameters, or a request body)
-   A structured response (usually JSON)
-   An HTTP status code indicating success or failure

When your frontend application sends a request to an endpoint, the server processes it and returns a response containing both data and a status code. Understanding how these elements work together is fundamental to integrating any REST API.

------------------------------------------------------------------------

## Key Concepts

### Endpoint Definition

An endpoint is a URL that represents a specific resource or collection of resources, such as:

-   `/users`
-   `/products/123`

It is the address your frontend uses to communicate with the backend.

------------------------------------------------------------------------

### HTTP Methods

HTTP methods define the action performed on the resource:

-   **GET** --- Retrieve data\
-   **POST** --- Create a new resource\
-   **PUT** --- Replace an existing resource\
-   **PATCH** --- Partially update an existing resource\
-   **DELETE** --- Remove a resource

The same URL path can behave differently depending on the method used.

Example:

    GET /users/123

retrieves a user, while:

    DELETE /users/123

removes that user.

------------------------------------------------------------------------

### Data Format

Most REST APIs send and receive data in **JSON (JavaScript Object Notation)** format. JSON is lightweight, readable, and easy to parse in JavaScript using built-in methods such as:

``` javascript
response.json()
```

------------------------------------------------------------------------

### Statelessness

REST APIs are stateless, meaning each request must contain all the information needed for the server to process it. The server does not store client session data between requests.

Authentication tokens (such as JWTs) are commonly included in request headers to verify identity without relying on server-side sessions.

------------------------------------------------------------------------

### HTTP Status Codes

Every response includes an HTTP status code that indicates the result of
the request.

Common examples include:

-   `200 OK` --- Request succeeded\
-   `400 Bad Request` --- Invalid request data\
-   `401 Unauthorized` --- Missing or invalid authentication\
-   `404 Not Found` --- Resource does not exist\
-   `500 Internal Server Error` --- Server-side issue

Frontend applications should always check the status code before
assuming a request succeeded.

------------------------------------------------------------------------

## Example: Retrieve a User

To see how these elements work together, consider an endpoint that
retrieves a single user by ID.

### Endpoint

    GET https://api.example.com/users/123

### What This Request Means

-   **Method:** `GET`\
-   **Resource:** `/users`\
-   **Path parameter:** `123` (the user's unique identifier)

This request asks the server to return data for the user with ID `123`.

------------------------------------------------------------------------

### Successful Response

**Status Code:** `200 OK`

``` json
{
  "id": 123,
  "name": "Jane Doe",
  "email": "jane@example.com",
  "created_at": "2026-01-10T14:32:00Z"
}
```

#### How to Interpret This Response

-   The `200` status code confirms the request succeeded.
-   The response body contains structured JSON data representing the user.
-   Each field is defined by the API and documented so developers know what to expect.

In a frontend application, you would parse this JSON and use it to render user data in the UI.

------------------------------------------------------------------------

### Error Example: User Not Found

If the user does not exist, the server may return:

**Status Code:** `404 Not Found`

``` json
{
  "error": "User not found"
}
```

#### What This Means

-   The `404` status code indicates the resource does not exist.
-   The response body provides additional context about the error.
-   Your frontend code should handle this case gracefully (for example, by displaying an error message).

------------------------------------------------------------------------

## Common Mistakes When Calling Endpoints

### Ignoring the HTTP Method

A URL alone does not define behavior. The HTTP method determines the action. Sending the wrong method may result in a `405 Method Not Allowed` error.

------------------------------------------------------------------------

### Not Checking the Status Code

Even when a response includes JSON, the request may have failed. Always verify the HTTP status code before processing the response data.

------------------------------------------------------------------------

### Sending Incorrect Request Data

For `POST`, `PUT`, or `PATCH` requests, ensure:

-   Required fields are included\
-   Data types are correct\
-   JSON is properly formatted

Invalid input often results in a `400 Bad Request` response.

------------------------------------------------------------------------

### Confusing Path and Query Parameters

Developers sometimes confuse:

    /users/123

with:

    /users?id=123

APIs document which format they expect. Using the wrong structure may lead to unexpected results.

------------------------------------------------------------------------

### Missing Authentication

If authentication headers (such as an API key or bearer token) are required but not included, the server may return `401 Unauthorized` or `403 Forbidden`.

------------------------------------------------------------------------

## Conclusion

An API endpoint is the fundamental unit of interaction in a REST API. It combines a URL path, an HTTP method, expected input, and structured responses to define how a client and server communicate.

By understanding how these components fit together, you can read API documentation more effectively and integrate endpoints into your frontend applications with confidence. Once you recognize the pattern---method, path, parameters, and response---working with additional endpoints becomes a repeatable and manageable process.
