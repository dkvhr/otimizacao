## Para rodar o programa

`python3 main.py < programa`

## Formato da entrada

`max` ou `min`
`n m` (n vars, m constraints)
vetor de custo (n numeros)
`m` linhas: n coeficientes, operador (`<=`, `>=` ou `=`) e valor do RHS

### Exemplo

```
max
2 3
3 2
1 2 <= 4
1 1 <= 2
1 -1 >= 1
```

Significa:

```
Maximize z = 3x1 + 2x2

sujeito a:
x1 + 2x2 <= 4
x1 + x2 <= 2
x1 - x2 >= 1
```
