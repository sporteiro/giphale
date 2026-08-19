# Development Rules

## Principles

1. **Maintain readability** - Code should be clear and easy to understand
2. **No icons in code** - Avoid using emojis or special characters in code
3. **No Spanish comments in code** - All code comments must be in English
4. **Reuse when possible** - Avoid code duplication, use existing abstractions
5. **Object-oriented programming** - Design using objects and classes
6. **Hexagonal architecture** - Separate domain logic from external dependencies
7. **TDD (Test Driven Development)** - Write tests before implementation
8. **DDD (Domain Driven Design)** - Focus on domain logic and bounded contexts
9. **Update requirements** - Always update requirements.txt when adding new dependencies

## Code Style Guidelines

- Use descriptive variable and function names
- Keep functions small and focused on single responsibility
- Follow SOLID principles
- Use type hints where appropriate
- Write meaningful commit messages
- Add logging for debugging and monitoring

## Testing Guidelines

- Write unit tests with mocks for external dependencies
- Write integration tests for real API calls (marked with @pytest.mark.integration)
- Run integration tests with: pytest -m integration
- Skip integration tests by default in CI/CD