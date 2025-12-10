"""
Apply transformations to BitLengthAware objects.
"""

from dataclasses import is_dataclass, replace

from localtypes import BitLengthAware, Primitive

from .operations import (
    Add,
    Delete,
    ExtendedOperation,
    Graft,
    Identity,
    Inner,
    Prune,
    Substitute,
)


class TransformationError(Exception):
    """Raised when a transformation cannot be applied."""

    pass


def apply_transformation(
    source: BitLengthAware, operation: ExtendedOperation
) -> BitLengthAware:
    """
    Applies a transformation to transform the source into the destination.

    Args:
        source: The source BitLengthAware object.
        operation: The operation to apply (Identity, Substitute, Prune, etc.).

    Returns:
        The transformed BitLengthAware object.
    """

    match operation.key.value:
        case None:
            return _apply_none_key(source, operation)
        case str() as field_name:
            return _apply_string_key(source, operation, field_name)
        case (str() as field_name, int() as index):
            return _apply_tuple_key(source, operation, field_name, index)
        case _:
            raise TransformationError(
                f"Invalid key {operation.key.value} for {operation}"
            )


def _apply_none_key(
    source: BitLengthAware, operation: ExtendedOperation
) -> BitLengthAware:
    """Handle operations with None key (root-level operations)."""
    match operation:
        case Identity(_):
            return source
        case Substitute(_, after):
            return after
        case Inner(_, operations) if isinstance(source, Primitive):
            for op in operations:
                if isinstance(op, Substitute) and op.key.value is None:
                    return op.after
            raise TransformationError(
                f"No applicable Substitute found in Inner for Primitive {source}"
            )
        case Inner(_, operations) if not is_dataclass(source) or isinstance(
            source, Primitive
        ):
            raise TransformationError(f"Inner operation on a non tree node: {source}")
        case Inner(_, operations):
            result = source
            for op in operations:
                result = apply_transformation(result, op)
            return result
        case _:
            raise TransformationError(f"Invalid operation for None key: {operation}")


def _apply_string_key(
    source: BitLengthAware, operation: ExtendedOperation, field_name: str
) -> BitLengthAware:
    """Handle operations with string key (field-level operations)."""
    if not is_dataclass(source) or isinstance(source, Primitive):
        raise TransformationError(f"String key operation on non tree node: {source}")

    if isinstance(operation, Graft):
        ancestor_field = getattr(operation.ancestor, field_name)
        return replace(
            operation.ancestor,
            **{field_name: apply_transformation(ancestor_field, operation.then)},
        )

    source_field = getattr(source, field_name)

    match operation:
        case Add(_, value) | Delete(_, value) if not isinstance(
            source_field, frozenset
        ):
            raise TransformationError(
                f"Add/Delete on non-frozenset field: {source_field} (key: {field_name})"
            )
        case Identity() | Substitute() | Inner() | Prune() | Graft() if not isinstance(
            source_field, BitLengthAware
        ):
            raise TransformationError(
                f"Operation on non-BitLengthAware field: {source_field} (key: {field_name})"
            )
        case Identity(_):
            return source
        case Substitute(_, after):
            return replace(source, **{field_name: after})
        case Inner(_, ops):
            assert isinstance(source_field, BitLengthAware)
            return replace(
                source,
                **{
                    field_name: apply_transformation(
                        source_field, Inner(operation.key.__class__(None), ops)
                    )
                },
            )
        case Prune(_, then):
            assert isinstance(source_field, BitLengthAware)
            return apply_transformation(source_field, then)
        case Add(_, value):
            assert isinstance(source_field, frozenset)
            return replace(
                source,
                **{field_name: source_field.union({value})},
            )
        case Delete(_, value):
            assert isinstance(source_field, frozenset)
            return replace(
                source,
                **{field_name: source_field.difference({value})},
            )
        case _:
            raise TransformationError(f"Invalid operation for string key: {operation}")


def _apply_tuple_key(
    source: BitLengthAware,
    operation: ExtendedOperation,
    field_name: str,
    index: int,
) -> BitLengthAware:
    """Handle operations with tuple key (indexed field operations)."""
    if not is_dataclass(source) or isinstance(source, Primitive):
        raise TransformationError(f"Tuple key operation on non tree node: {source}")

    if isinstance(operation, Graft):
        ancestor_field = getattr(operation.ancestor, field_name)
        return replace(
            operation.ancestor,
            **{
                field_name: ancestor_field[:index]
                + (apply_transformation(source, operation.then),)
                + ancestor_field[index + 1 :]
            },
        )

    source_field = getattr(source, field_name)
    assert isinstance(source_field, tuple)

    match operation:
        case Identity(_):
            return source
        case Substitute(_, after):
            return replace(
                source,
                **{
                    field_name: source_field[:index]
                    + (after,)
                    + source_field[index + 1 :]
                },
            )
        case Inner(_, ops):
            return replace(
                source,
                **{
                    field_name: source_field[:index]
                    + (
                        apply_transformation(
                            source_field[index],
                            Inner(operation.key.__class__(None), ops),
                        ),
                    )
                    + source_field[index + 1 :]
                },
            )
        case Add(_, value):
            return replace(
                source,
                **{field_name: source_field + (value,)},
            )
        case Delete(_, value) if (
            index == len(source_field) - 1 and source_field[-1] == value
        ):
            return replace(source, **{field_name: source_field[:-1]})
        case Delete(_, value):
            raise TransformationError(
                f"Delete at index {index} invalid: expected last element {value}"
            )
        case Prune(_, then):
            assert isinstance(source_field[index], BitLengthAware)
            return apply_transformation(source_field[index], then)
        case _:
            raise TransformationError(f"Invalid operation for tuple key: {operation}")
